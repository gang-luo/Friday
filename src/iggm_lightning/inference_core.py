# -*- coding: utf-8 -*-
"""Shared pieces between the Lightning validation path and scripts/inference.py.

Everything here already existed, inlined in one of the two callers.  It was
pulled out so there is exactly one definition of "how a noisy state becomes
network inputs" and "how a training checkpoint becomes a runnable model" -- the
two places where a silent divergence between training and inference would be
hardest to notice and most damaging.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch import nn

from IgGM.model import DesignModel


def build_model_inputs(
    plm_featurizer: nn.Module,
    prot_data_pert: Dict[str, Any],
    prot_data_curr: Dict[str, Any],
) -> Dict[str, Any]:
    """Turn one noisy state into network inputs.

    Used by training (via IgGMLightningModule._featurize_pert), by the precise
    validation layer, and by scripts/inference.py, so all three feed the network
    an identically-constructed tensor.
    """
    with torch.no_grad():
        inputs = DesignModel.featurize(plm_featurizer, prot_data_pert)

    if prot_data_curr.get("contact") is None:
        ic_feat = torch.zeros_like(prot_data_curr["asym_id"])
        ag_len = len(prot_data_curr["epitope"])
        ic_feat[:, -ag_len:] = prot_data_curr["epitope"]
        inputs["ic_feat"] = ic_feat.unsqueeze(-1).type_as(inputs["sfea-i"])
    else:
        bs, length = prot_data_curr["asym_id"].shape
        ic_feat = torch.zeros(bs, length, length, device=prot_data_curr["asym_id"].device)
        ic_feat[:, ...] = prot_data_curr["contact"]
        inputs["ic_feat"] = ic_feat.unsqueeze(-1).type_as(inputs["sfea-i"])
    return inputs


def load_design_state_dict(
    ckpt_path: str, use_ema: bool = True
) -> tuple[Dict[str, torch.Tensor], str]:
    """Extract the DesignModel weights from a Lightning checkpoint.

    `use_ema` defaults to True and that default matters: validation runs on the
    EMA weights (IgGMLightningModule.on_validation_start), so the checkpoint that
    ModelCheckpoint kept was selected by a metric the EMA weights earned.  Loading
    `state_dict` instead would run inference on a DIFFERENT set of weights than
    the ones the selection was based on -- quietly, with no error anywhere.

    Returns (state_dict, which) where `which` is "ema" or "raw", so the caller can
    print what it actually loaded rather than what it asked for.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw = _strip_model_prefix(ckpt.get("state_dict", ckpt))

    shadow = ckpt.get("ema_shadow") if use_ema else None
    if not shadow:
        return raw, "raw"

    # The shadow is keyed by DesignModel's own parameter names (it is built from
    # self.model.state_dict()), so unlike `state_dict` it needs no prefix
    # stripping.  It only covers floating-point tensors, so the integer buffers
    # still have to come from the raw state_dict underneath -- hence overlay
    # rather than replace.
    merged = dict(raw)
    n_applied = 0
    for k, v in shadow.items():
        if k in merged:
            merged[k] = v
            n_applied += 1
    if n_applied == 0:
        raise RuntimeError(
            f"{ckpt_path}: ema_shadow has {len(shadow)} entries but none matched "
            f"the {len(raw)} model keys -- refusing to silently fall back to the "
            f"non-EMA weights.  Pass --no_ema if that is what you want."
        )
    return merged, f"ema ({n_applied}/{len(raw)} tensors)"


def _strip_model_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop the LightningModule's `model.` prefix; pass through if absent."""
    if any(k.startswith("model.") for k in state_dict):
        return {
            k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")
        }
    return dict(state_dict)


@torch.no_grad()
def run_reverse_sampling(
    *,
    model: nn.Module,
    plm_featurizer: nn.Module,
    diffuser: Any,
    prot_data_curr: Dict[str, Any],
    n_sample_steps: Optional[int] = None,
    seed: Optional[int] = None,
    chunk_size: Optional[int] = None,
    seq_feedback: bool = True,
    return_trajectory: bool = False,
) -> Dict[str, Any]:
    """Reverse-sample one sample.  Same call shape the precise layer uses."""

    def forward_fn(prot_data_pert: Dict[str, Any]) -> Dict[str, Any]:
        inputs = build_model_inputs(plm_featurizer, prot_data_pert, prot_data_curr)
        return model(inputs, inputs_addi=None, chunk_size=chunk_size)

    return diffuser.reverse_sample(
        prot_data_curr,
        forward_fn,
        n_sample_steps=n_sample_steps,
        seed=seed,
        seq_feedback=seq_feedback,
        return_trajectory=return_trajectory,
    )
