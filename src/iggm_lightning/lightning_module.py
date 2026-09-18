# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""PyTorch Lightning wrapper for IgGM training.

This module maps the legacy IgGM train/eval forward path into Lightning hooks
without changing diffusion equations, noise schedules, sampling process, or the
DesignModel forward implementation.
"""

from __future__ import annotations

import hashlib
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import nn

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

import torch.distributed as dist

from IgGM.model import DesignModel
from IgGM.protein.prot_constants import RESD_NAMES_1C
from .losses import IgGMLossConfig,IgGMPaperLoss
from .atom14_sync import Atom14SeqSync
from .inference_core import build_model_inputs
from .metrics import MetricConfig, StructureMetrics


def mem(tag):
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    p = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{tag}] alloc={a:.2f} GB reserved={r:.2f} GB peak={p:.2f} GB")

@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999) # 0.9 / 0.5
    eps: float = 1e-8


@dataclass
class StageTrainingConfig:
    """Two-phase training controls aligned with paper-style training."""

    stage1_epochs: int = 0
    stage2_enable_seq_recovery: bool = True
    stage2_mix_weights: Dict[str, int] | None = None

    def __post_init__(self):
        if self.stage2_mix_weights is None:
            # default ratio: CDR-H3 : CDR-H1 : CDR-H2 : all-CDR = 4:2:2:2
            self.stage2_mix_weights = {
                "cdr_h3": 4,
                "cdr_h1": 2,
                "cdr_h2": 2,
                "cdr_all": 2,
            }


class ModelEMA:
    """Minimal EMA for optional shadow parameter tracking."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _align_device(self, model: nn.Module) -> None:
        """Move the shadow onto the model's current device if it has moved.

        This EMA is constructed in LightningModule.__init__, which runs BEFORE
        Lightning transfers the model to the accelerator, so the shadow starts
        life on CPU while the live weights end up on cuda:0.  Without this the
        first update() raises "Expected all tensors to be on the same device".
        Resuming from a checkpoint can also restore the shadow onto a different
        device than the current run uses, so this is re-checked every call --
        it is a device comparison per tensor, not a copy, once aligned.
        """
        msd = model.state_dict()
        for k, v in self.shadow.items():
            tgt = msd.get(k)
            if tgt is not None and (v.device != tgt.device or v.dtype != tgt.dtype):
                self.shadow[k] = v.to(device=tgt.device, dtype=tgt.dtype)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self._align_device(model)
        msd = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Swap the EMA weights in, stashing the live ones for restore()."""
        self._align_device(model)
        msd = model.state_dict()
        self._backup = {k: msd[k].detach().clone() for k in self.shadow}
        for k, v in self.shadow.items():
            msd[k].copy_(v)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Put the live training weights back after an EMA-evaluated pass."""
        if not self._backup:
            return
        msd = model.state_dict()
        for k, v in self._backup.items():
            msd[k].copy_(v)
        self._backup = {}


class IgGMLightningModule(pl.LightningModule):
    """Lightning mapping of IgGM's original diffusion training skeleton."""

    def __init__(
        self,
        model: nn.Module,
        plm_featurizer: nn.Module,
        diffuser: nn.Module,
        *,
        optimizer_cfg: Optional[OptimizerConfig] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        grad_clip_val: Optional[float] = 1.0,
        use_amp: bool = True,
        ema_decay: Optional[float] = None,
        debug_shapes: bool = False,
        loss_cfg: Optional[IgGMLossConfig] = None,
        metric_cfg: Optional[MetricConfig] = None,
        stage_cfg: Optional[StageTrainingConfig] = None,
        precise_eval_every_n_val: int = 0,
        precise_eval_steps: Optional[int] = None,
        precise_eval_seed: int = 20260909,
        cheap_eval_seed: int = 20260915,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be torch.nn.Module")
        self.model = model
        self.plm_featurizer = plm_featurizer
        self.diffuser = diffuser
        for param in self.plm_featurizer.parameters():
            param.requires_grad = False
        self.plm_featurizer.eval()
        self.optimizer_cfg = optimizer_cfg or OptimizerConfig()
        self.scheduler_cfg = scheduler_cfg or {}
        self.grad_clip_val = grad_clip_val
        self.enable_amp = use_amp
        self.debug_shapes = debug_shapes
        self._shape_printed = False
        self.ema = ModelEMA(self.model, ema_decay) if ema_decay is not None else None
        self.loss_fn = IgGMPaperLoss(loss_cfg)
        self.metric_fn = StructureMetrics(metric_cfg)
        self.stage_cfg = stage_cfg or StageTrainingConfig()
        self.atom14_sync = Atom14SeqSync()
        self._skip_optimizer_step_due_to_oom = False
        # A4: prob of applying training-time self-conditioning per step (0 disables).
        self.self_cond_prob = 0.0 # 0.5

        
        self.register_buffer("_rota_pred_energy_ema", torch.tensor(0.0), persistent=False)
        self.register_buffer("_rota_target_energy_ema", torch.tensor(0.0), persistent=False)
        self.register_buffer("_rota_dot_ema", torch.tensor(0.0), persistent=False)
        # AB1: absorption = 1 - residual_angle / target_angle.  The two angles
        # are EMA'd separately (not the ratio) so a near-zero target angle on a
        # single step cannot blow the metric up.
        self.register_buffer("_rota_target_angle_ema", torch.tensor(0.0), persistent=False)
        self.register_buffer("_rota_residual_angle_ema", torch.tensor(0.0), persistent=False)
        self._rota_diag_initialized = False

        # --- precise (reverse-sampling) validation layer ---------------------
        # Cadence is counted in VALIDATION RUNS, not epochs: check_val_every_n_epoch
        # is a separate knob, and expressing this in epochs silently changes the
        # number of precise evals whenever that knob moves.  0 disables the layer.
        self.precise_eval_every_n_val = int(precise_eval_every_n_val)
        self.precise_eval_steps = precise_eval_steps  # None = full 200-step grid
        self.precise_eval_seed = int(precise_eval_seed)
        self.cheap_eval_seed = int(cheap_eval_seed)
        self._val_run_counter = 0
        self._pending_precise_batches: List[Dict[str, Any]] = []
        self._last_precise_metrics: Dict[str, float] = {}

    @staticmethod
    def _ddp_any_true(flag: bool) -> bool:
        """Synchronize boolean failure flags across ranks for DDP-safe fallbacks."""
        if not (dist.is_available() and dist.is_initialized()):
            return bool(flag)
        val = torch.tensor([1 if flag else 0], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.int32)
        dist.all_reduce(val, op=dist.ReduceOp.MAX)
        return bool(val.item() > 0)

    def _zero_loss(self) -> torch.Tensor:
        """Build a graph-safe zero loss to skip optimizer update without crashing."""
        try:
            param = next(self.model.parameters())
            return param.sum() * 0.0
        except StopIteration:
            return torch.zeros((), device=self.device, requires_grad=True)

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda error: out of memory" in msg

    def _move_to_device(self, obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.to(self.device)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move_to_device(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(v) for v in obj)
        return obj

    def _is_stage2(self) -> bool:
        return self.current_epoch >= int(self.stage_cfg.stage1_epochs)

    @staticmethod
    def _cdr_mask_from_payload(payload: Dict[str, Any], mode: str) -> Optional[List[int]]:
        cdr = payload.get("cdr_sequences") or {}
        seq_lens = payload.get("sequence_lengths") or {}
        h_len = int(seq_lens.get("H", 0))
        l_len = int(seq_lens.get("L", 0))

        def _offset_indices(keys: List[str], offset: int) -> List[int]:
            out: List[int] = []
            for key in keys:
                for idx in cdr.get(key, []):
                    ii = int(idx) - 1 + offset
                    if ii >= offset:
                        out.append(ii)
            return out

        h1 = _offset_indices(["cdr_H1"], 0)
        h2 = _offset_indices(["cdr_H2"], 0)
        h3 = _offset_indices(["cdr_H3"], 0)
        l_all = _offset_indices(["cdr_L1", "cdr_L2", "cdr_L3"], h_len)

        if mode == "cdr_h1":
            return h1 or None
        if mode == "cdr_h2":
            return h2 or None
        if mode == "cdr_h3":
            return h3 or None
        if mode == "cdr_all":
            all_idx = sorted(set(h1 + h2 + h3 + l_all))
            return all_idx or None
        return None

    def _apply_stage_mask(self, prot_data_curr: Dict[str, Any], payload: Dict[str, Any]) -> None:
        if not self._is_stage2():
            return
        mix = self.stage_cfg.stage2_mix_weights or {}
        keys = [k for k, w in mix.items() if int(w) > 0]
        if not keys:
            return
        weights = [int(mix[k]) for k in keys]
        mode = random.choices(keys, weights=weights, k=1)[0]
        idxs = self._cdr_mask_from_payload(payload, mode)
        if not idxs:
            return
        mask_design = torch.zeros_like(prot_data_curr["mask_design"])
        valid = [i for i in idxs if 0 <= i < mask_design.shape[0]]
        if not valid:
            return
        mask_design[valid] = 1
        prot_data_curr["mask_design"] = mask_design

    def _build_inputs_cm(
        self,
        prot_data_curr: Dict[str, Any],
        idx_step: int,
        fixed_noise_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        if fixed_noise_seed is None:
            prot_data_pert = self.diffuser.run(prot_data_curr, idx_step)
        else:
            prot_data_pert = self.diffuser.run_fixed(
                prot_data_curr, idx_step, fixed_noise_seed
            )
        return self._featurize_pert(prot_data_pert, prot_data_curr)

    def _featurize_pert(
        self, prot_data_pert: Dict[str, Any], prot_data_curr: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Turn one noisy state into network inputs.

        Split out of _build_inputs_cm so reverse sampling -- which produces its own
        noisy state rather than calling diffuser.run -- goes through the exact same
        featurization and ic_feat construction as training.  The body now lives in
        inference_core so scripts/inference.py shares the single definition.
        """
        return build_model_inputs(self.plm_featurizer, prot_data_pert, prot_data_curr)

    def _compute_loss(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self.loss_fn(inputs, outputs)

    # Step at which _log_grad_balance samples the gradients.  Late enough that
    # coord_head has left its zero init (see that method's docstring), early
    # enough to still act on the reading.
    _GRAD_BALANCE_STEP = 50

    def _log_grad_balance(self, loss_dict):
        """Print, once, the gradient each CDR loss term puts on coord_head.

        Why: on 2026-09-01 loss_bond was enabled at weight 1.0 and fitted
        aar_cdr collapsed 0.83 -> 0.11 while bond length itself improved 6x.
        The loss VALUES gave no warning (bond was 0.001, smaller than loss_cdr).
        The cause was the gradient: loss_cdr is an MSE in x0_norm space (it
        divides by cdr_scale**2) while loss_bond was in physical A**2, so at
        cdr_scale=6 the bond term hit coord_head 36x harder.

        Loss values cannot predict this in general -- smooth_lddt is bounded in
        [0,1] with saturating sigmoids, so it can show a large value and
        contribute almost no gradient.  Hence: measure, do not estimate.

        `ratio` is the number that decides who wins; `w_balanced` is the weight
        that would equalize that term's gradient with loss_cdr's.

        Measured at step _GRAD_BALANCE_STEP, not at step 0: coord_head is
        zero-initialized, so at step 0 every atom is predicted at cdr_mu, every
        interatomic distance is 0, and any distance-based term (bond,
        smooth_lddt) has identically zero gradient at that degenerate point.
        loss_cdr is unaffected because it compares absolute coordinates.
        """
        step = int(self.global_step)
        if getattr(self, "_grad_balance_logged", False) or step < self._GRAD_BALANCE_STEP:
            return
        self._grad_balance_logged = True

        head = getattr(getattr(self.model, "cdr_loop_head", None), "coord_head", None)
        if head is None:
            for mod in self.model.modules():
                if hasattr(mod, "coord_head"):
                    head = mod.coord_head
                    break
        if head is None or head.weight is None:
            print("[GradBalance] coord_head not found; skipped", flush=True)
            return

        names = ["loss_cdr", "loss_bond", "loss_smooth_lddt", "loss_seq"]
        norms = {}
        for n in names:
            t = loss_dict.get(n)
            if not (torch.is_tensor(t) and t.requires_grad):
                continue
            try:
                g = torch.autograd.grad(
                    t, head.weight, retain_graph=True, allow_unused=True
                )[0]
            except RuntimeError as exc:
                print(f"[GradBalance] {n}: grad failed ({exc})", flush=True)
                continue
            norms[n] = 0.0 if g is None else float(g.detach().norm())

        base = norms.get("loss_cdr")
        print("[GradBalance] gradient on coord_head.weight (once, first step):",
              flush=True)
        for n, v in norms.items():
            val = loss_dict.get(n)
            val = float(val) if torch.is_tensor(val) else float("nan")
            if base and base > 0:
                ratio = v / base
                wb = (1.0 / ratio) if ratio > 0 else float("inf")
                print(f"    {n:18s} value={val:10.4g}  |g|={v:10.4g}  "
                      f"ratio={ratio:8.3f}  w_balanced={wb:9.4g}", flush=True)
            else:
                print(f"    {n:18s} value={val:10.4g}  |g|={v:10.4g}", flush=True)

    @staticmethod
    def _decode_pred_seq(logits_1d: torch.Tensor) -> str:
        if logits_1d.ndim != 2:
            raise ValueError(f"Unexpected sequence logit rank: {tuple(logits_1d.shape)}")
        # Support both [L, C] and [C, L] layouts from different model checkpoints.
        if logits_1d.shape[0] == len(RESD_NAMES_1C) and logits_1d.shape[1] != len(RESD_NAMES_1C):
            logits_1d = logits_1d.transpose(0, 1)
        token_ids = logits_1d.argmax(dim=-1).detach().cpu().tolist()
        return ''.join(RESD_NAMES_1C[i] for i in token_ids)

    @staticmethod
    def _safe_log_name(text: str) -> str:
        return str(text).strip().replace("/", "_").replace(" ", "_")

    @staticmethod
    def _is_eval_stage(stage: str) -> bool:
        return stage == "val" or stage.startswith("test")

    def _shared_step(
        self,
        batch: Dict[str, Any],
        stage: str,
        *,
        log_prefix: Optional[str] = None,
        compute_metrics: Optional[bool] = None,
        compute_cheap_metrics: bool = False,
        fixed_noise_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Run one single-step denoising pass and log it.

        `log_prefix` names the logged scalars independently of `stage`, so the
        cheap validation layer can report each pinned timestep separately
        (val_t25/..., val_t175/...) while every one of them still behaves as a
        "val" stage internally.  `compute_metrics` overrides the default
        (metrics on for eval stages): the cheap layer sets it False because
        StructureMetrics on a single-step prediction is expensive and is not the
        number we select checkpoints on -- that comes from the precise layer.
        """
        prefix = log_prefix or stage
        if compute_metrics is None:
            compute_metrics = self._is_eval_stage(stage)
        idx_step = int(batch["idx_step"])
        # # train: importance-sample step (log-normal sigma) instead of dataloader's
        # # uniform pick, to concentrate on the high-info SNR~1 band.
        # if stage == "train" and hasattr(self.diffuser, "sample_step"):
        #     idx_step = self.diffuser.sample_step()
        payload = batch.get("payload")
        if payload is None:
            raise RuntimeError("Dataset must provide resolved `payload` for lazy loading.")
        prot_data_curr = self._move_to_device(payload["prot_data_curr"])

        if stage == "train":
            self._apply_stage_mask(prot_data_curr, payload)
        inputs_addi = batch.get("inputs_addi")

        local_fail = False
        inputs = outputs = loss_dict = None

        # try:
        inputs = self._build_inputs_cm(
            prot_data_curr, idx_step, fixed_noise_seed=fixed_noise_seed
        )

        # A4: training-time self-conditioning (Chen et al. 2022). With prob
        # self_cond_prob, run one no-grad forward to get x0_hat, then feed it back
        # as conditioning (step all-zeros mode). The other fraction trains the
        # cold-start path (inputs_addi=None) so inference without prior still works.
        if (inputs_addi is None and stage == "train"
                and self.self_cond_prob > 0.0 and random.random() < self.self_cond_prob):
            with torch.no_grad():
                out_sc = self.model(inputs, inputs_addi=None, chunk_size=batch.get("chunk_size"))
            inputs_addi = {
                "step": [0],
                "sfea": out_sc["sfea"].detach(),
                "pfea": out_sc["pfea"].detach(),
                "cord": out_sc["3d"]["cord"][-1].detach(),
            }

        outputs = self.model(inputs, inputs_addi=inputs_addi, chunk_size=batch.get("chunk_size"))
        loss_dict = self._compute_loss(inputs, outputs)
        if stage == "train":
            self._log_grad_balance(loss_dict)

        self.log(f"{prefix}/loss", loss_dict["loss"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_backbone", loss_dict["loss_backbone"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_cdr", loss_dict["loss_cdr"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_cdr_backbone", loss_dict["loss_cdr_backbone"], prog_bar=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_cdr_sidechain", loss_dict["loss_cdr_sidechain"], prog_bar=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_cdr_virtual", loss_dict["loss_cdr_virtual"], prog_bar=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
        # self.log(f"{prefix}/loss_viol", loss_dict["loss_viol"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_smooth_lddt", loss_dict["loss_smooth_lddt"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_bond", loss_dict["loss_bond"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_bond_backbone", loss_dict["loss_bond_backbone"], prog_bar=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_bond_sidechain", loss_dict["loss_bond_sidechain"], prog_bar=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_trsl", loss_dict["loss_trsl"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_rota", loss_dict["loss_rota"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        # self.log(f"{prefix}/w_cdr", loss_dict["w_cdr"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_trsl_residual", loss_dict["loss_trsl_residual"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_rota_residual", loss_dict["loss_rota_residual"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}/loss_seq", loss_dict["loss_seq"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)


        if stage == "train":
            diag = loss_dict["rotation_diag"]
            decay = 0.95

            if not self._rota_diag_initialized:
                self._rota_pred_energy_ema.copy_(diag["pred_energy"])
                self._rota_target_energy_ema.copy_(diag["target_energy"])
                self._rota_dot_ema.copy_(diag["dot"])
                self._rota_target_angle_ema.copy_(diag["target_angle"])
                self._rota_residual_angle_ema.copy_(diag["residual_angle"])
                self._rota_diag_initialized = True
            else:
                self._rota_pred_energy_ema.lerp_(diag["pred_energy"], 1.0 - decay)
                self._rota_target_energy_ema.lerp_(diag["target_energy"], 1.0 - decay)
                self._rota_dot_ema.lerp_(diag["dot"], 1.0 - decay)
                self._rota_target_angle_ema.lerp_(diag["target_angle"], 1.0 - decay)
                self._rota_residual_angle_ema.lerp_(diag["residual_angle"], 1.0 - decay)

            eps = 1e-8
            pred_energy = self._rota_pred_energy_ema
            target_energy = self._rota_target_energy_ema
            dot = self._rota_dot_ema

            norm_ratio = torch.sqrt(
                (pred_energy + eps) / (target_energy + eps)
            )
            energy_cosine = dot / torch.sqrt(
                (pred_energy * target_energy).clamp_min(eps)
            )
            energy_gain = (
                2.0 * dot - pred_energy
            ) / target_energy.clamp_min(eps)

            # AB1: absorption in degrees-free form.  1.0 = the model applied
            # exactly the required correction, 0.0 = it did not move at all,
            # negative = it made things worse.
            rota_absorption = 1.0 - (
                self._rota_residual_angle_ema
                / self._rota_target_angle_ema.clamp_min(1e-6)
            )
            self.log(
                "train/rota_absorption", rota_absorption,
                on_step=True, on_epoch=False, prog_bar=False,
                sync_dist=True, add_dataloader_idx=False,
            )
            self.log(
                "train/rota_target_angle_deg",
                self._rota_target_angle_ema * (180.0 / math.pi),
                on_step=True, on_epoch=False, prog_bar=False,
                sync_dist=True, add_dataloader_idx=False,
            )
            self.log(
                "train/rota_residual_angle_deg",
                self._rota_residual_angle_ema * (180.0 / math.pi),
                on_step=True, on_epoch=False, prog_bar=False,
                sync_dist=True, add_dataloader_idx=False,
            )

            self.log(
                "train/rota_norm_ratio", norm_ratio,
                on_step=True, on_epoch=False, prog_bar=False,
                sync_dist=True, add_dataloader_idx=False,
            )
            self.log(
                "train/rota_energy_cosine", energy_cosine,
                on_step=True, on_epoch=False, prog_bar=False,
                sync_dist=True, add_dataloader_idx=False,
            )
            self.log(
                "train/rota_energy_gain", energy_gain,
                on_step=True, on_epoch=False, prog_bar=False,
                sync_dist=True, add_dataloader_idx=False,
            )
            
        # self.log(f"{prefix}/loss_closure", loss_dict["loss_closure"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        # self.log(f"{prefix}/loss_marker_topology", loss_dict["loss_marker_topology"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        # self.log(f"{prefix}/loss_marker_count", loss_dict["loss_marker_count"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        # self.log(f"{prefix}/loss_marker_aar", loss_dict["loss_marker_aar"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)

        # except Exception as exc:
        #     local_fail = True

        if compute_cheap_metrics:
            pred_cord = outputs["3d"]["cord"][-1][0]
            tgt_cord = inputs["cord-o"]
            if tgt_cord.ndim == 4:
                tgt_cord = tgt_cord[0]
            true_seq = payload.get("seq_true", inputs["seq-o"][0])
            pred_seq = self.atom14_sync.decode_cdr_sequence(
                seq_true=true_seq,
                pred_cord_n14_tf=pred_cord,
                pred_cmsk_n14_tf=inputs.get("cmsk_atom14", inputs["cmsk-p"]),
                cdr_mask=inputs["cdr_mask"],
            )
            loop_metrics = self.metric_fn.compute_loop_metrics(
                pred_cord,
                tgt_cord,
                pred_seq,
                true_seq,
                (payload.get("cdr_sequences") or {}).get("cdr_H3", []),
                (payload.get("cdr_sequences") or {}),
                (payload.get("sequence_lengths") or {}),
            )
            for key in ("aar_loop_mean", "rmsd_loop_mean"):
                self.log(
                    f"{prefix}/{key}",
                    loop_metrics[key],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    add_dataloader_idx=False,
                )

        if compute_metrics:
            pred_cord = outputs["3d"]["cord"][-1][0]
            tgt_cord = inputs["cord-o"]
            if tgt_cord.ndim == 4:
                tgt_cord = tgt_cord[0]
            true_seq = payload.get("seq_true", inputs["seq-o"][0])
            pred_seq = self.atom14_sync.decode_cdr_sequence(
                seq_true=true_seq,
                pred_cord_n14_tf=pred_cord,
                pred_cmsk_n14_tf=inputs.get("cmsk_atom14", inputs["cmsk-p"]),
                cdr_mask=inputs["cdr_mask"],
            )
            cdr_h3 = (payload.get("cdr_sequences") or {}).get("cdr_H3", [])
            metric_dict = self.metric_fn(
                pred_cord,
                tgt_cord,
                pred_seq,
                true_seq,
                cdr_h3,
                asym_id=inputs.get("asym-id"),
                cdr_sequences=(payload.get("cdr_sequences") or {}),
                seq_lengths=(payload.get("sequence_lengths") or {}),
                native_atom_mask=prot_data_curr["cmsk"],
                antibody_mask=prot_data_curr["mask_ab"],
                antigen_mask=prot_data_curr["antigen_mask"],
                cdr_mask=prot_data_curr["cdr_mask"],
            )
            for k, v in metric_dict.items():
                self.log(f"{prefix}/{k}", v, prog_bar=(k == "tm_score"), on_step=False, on_epoch=True,add_dataloader_idx=False,)

        global_fail = self._ddp_any_true(local_fail)
        loss_flag = loss_dict["loss"] if loss_dict is not None else None
        local_nonfinite = bool(loss_flag is not None and (not torch.isfinite(loss_flag.detach()).item()))
        global_nonfinite = self._ddp_any_true(local_nonfinite)
        if global_nonfinite or global_fail:
            self.log(f"{prefix}/skip_failed_batch or skip_nonfinite_loss", torch.tensor(1.0, device=self.device), prog_bar=False, on_step=(stage == "train"), on_epoch=True, batch_size=1, add_dataloader_idx=False)
            return self._zero_loss() if stage == "train" else None

        return loss_dict["loss"]

    # ------------------------------------------------------------------
    # Precise validation layer: full reverse sampling + StructureMetrics
    # ------------------------------------------------------------------

    def _precise_eval_due(self) -> bool:
        if self.precise_eval_every_n_val <= 0:
            return False
        if self._is_sanity_checking():
            return False
        # _val_run_counter is incremented in on_validation_start, so it is already
        # 1-based here: run 1 evaluates, then every Nth after it.
        return (self._val_run_counter - 1) % self.precise_eval_every_n_val == 0

    @torch.no_grad()
    def _run_precise_eval(self) -> None:
        """Reverse-sample every pending validation sample and report metric MEANS.

        Runs one sample at a time because the whole pipeline is batch-1 (the
        collate_fn keeps only batch[0] and no tensor carries a batch dimension),
        so there is no padding path to exercise here.

        Each sample gets its own fixed seed = precise_eval_seed + index, so the
        entire trajectory (translation noise, CDR noise, IGSO(3) rotations) is
        identical across checkpoints and two runs differ only by the weights.
        """
        batches = self._pending_precise_batches
        self._pending_precise_batches = []
        if not batches:
            return

        was_training = self.model.training
        self.model.eval()
        accum: Dict[str, List[float]] = {}
        n_ok = 0

        for idx, batch in enumerate(batches):
            payload = batch.get("payload")
            if payload is None:
                continue
            prot_data_curr = self._move_to_device(payload["prot_data_curr"])

            def forward_fn(prot_data_pert, _curr=prot_data_curr, _batch=batch):
                inputs = self._featurize_pert(prot_data_pert, _curr)
                return self.model(
                    inputs, inputs_addi=None, chunk_size=_batch.get("chunk_size")
                )

            result = self.diffuser.reverse_sample(
                prot_data_curr,
                forward_fn,
                n_sample_steps=self.precise_eval_steps,
                seed=self.precise_eval_seed + idx,
            )

            pred_cord = result["cord"]
            tgt_cord = prot_data_curr["cords_atom14"].float()
            if tgt_cord.ndim == 4:
                tgt_cord = tgt_cord[0]
            true_seq = payload.get("seq_true", prot_data_curr["seq"])
            cmsk14 = prot_data_curr["cmsk_atom14"]
            pred_seq = self.atom14_sync.decode_cdr_sequence(
                seq_true=true_seq,
                pred_cord_n14_tf=pred_cord,
                pred_cmsk_n14_tf=cmsk14,
                cdr_mask=prot_data_curr["cdr_mask"],
            )
            asym_id = prot_data_curr["asym_id"]
            if asym_id.ndim == 1:
                asym_id = asym_id.unsqueeze(0)
            metric_dict = self.metric_fn(
                pred_cord,
                tgt_cord,
                pred_seq,
                true_seq,
                (payload.get("cdr_sequences") or {}).get("cdr_H3", []),
                asym_id=asym_id,
                cdr_sequences=(payload.get("cdr_sequences") or {}),
                seq_lengths=(payload.get("sequence_lengths") or {}),
                native_atom_mask=prot_data_curr["cmsk"],
                antibody_mask=prot_data_curr["mask_ab"],
                antigen_mask=prot_data_curr["antigen_mask"],
                cdr_mask=prot_data_curr["cdr_mask"],
            )
            for k, v in metric_dict.items():
                accum.setdefault(k, []).append(float(v))
            n_ok += 1

            if idx == 0:
                print(
                    f"[Precise] reverse sampling: {len(result['schedule']) - 1} network "
                    f"evaluations, t from {result['schedule'][0]} down to "
                    f"{result['final_step']}",
                    flush=True,
                )

        if was_training:
            self.model.train()
        if n_ok == 0:
            return

        self._last_precise_metrics = {
            k: sum(v) / len(v) for k, v in accum.items()
        }
        summary = "  ".join(
            f"{k}={v:.4f}" for k, v in sorted(self._last_precise_metrics.items())
        )
        print(f"[Precise] n={n_ok}  {summary}", flush=True)

    def _log_precise_metrics(self) -> None:
        """Log the precise metrics on EVERY validation run, not just precise ones.

        ModelCheckpoint raises (not warns) when its monitored key is absent from a
        validation run's metrics -- see model_checkpoint.py `_save_topk_checkpoint`,
        which only downgrades to a warning before the val loop has ever run.  Since
        the precise layer fires every Nth run, the monitor would be missing on the
        other N-1 and kill the job.

        So the last computed values are carried forward.  This is safe for
        selection: mode="max" compares with a strict `>`, so a repeated value can
        never displace the checkpoint that originally earned it.
        """
        for k, v in self._last_precise_metrics.items():
            self.log(
                f"val/precise_{k}",
                torch.tensor(float(v), device=self.device),
                prog_bar=(k == "tm_score"),
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )

    def on_fit_start(self) -> None:
        self._log_stage_snapshot()

    def _log_stage_snapshot(self) -> None:
        """Print, once at fit start, every knob that defines WHICH STAGE this run is.

        Why this exists: "which stage am I in" has no single representation in the
        code -- it is spread over a hardcoded override in Diffuser.run, the
        _bucket_timestep folding, two manual_seed calls, and the pinned timesteps
        in validation_step.  Entering the next stage means editing several places
        in two files, and missing any one of them yields a run that LOOKS right
        but is not the experiment you think it is.  That exact failure already
        cost this project a 1000-step run (see _log_active_loss_terms) and a set
        of misaligned probes.

        Same tactic as _log_active_loss_terms: values are parsed out of the live
        source text rather than duplicated here, so this banner cannot drift away
        from the lines you actually edit.  Anything unparseable prints as "?" --
        a "?" means go read the code, not that the knob is off.
        """
        import inspect
        import re

        def _src(obj):
            try:
                return inspect.getsource(obj)
            except (OSError, TypeError):
                return ""

        def _live(src):
            """Drop commented-out code before matching.

            Without this the banner reports DEAD code as active: a commented-out
            `# idxs_step = self._bucket_timestep(...)` or
            `# self.rota_sampler.generator.manual_seed(56)` still matches a naive
            regex, so the banner claims the stage knob is on when the user has
            just turned it off.  That is the exact failure this banner exists to
            prevent, so the stripping happens before every match below.
            """
            out = []
            for line in src.splitlines():
                code = line.split("#", 1)[0]
                if code.strip():
                    out.append(code)
            return "\n".join(out)

        lines = ["[Stage] ---- stage snapshot (parsed from live source) ----"]

        # 1. Pinned training timestep: the LAST uncommented `idxs_step = <int>`
        #    assignment in Diffuser.run wins, since it overrides what came before.
        run_src = _live(_src(type(self.diffuser).run))
        pinned = re.findall(
            r"^\s*idxs_step\s*=\s*(\d+)\s*$", run_src, flags=re.MULTILINE
        )
        if pinned:
            lines.append(
                f"[Stage] train timestep : PINNED to {pinned[-1]} "
                f"(hardcoded override in Diffuser.run) -- dataset's random t is discarded"
            )
        else:
            n_steps = getattr(self.diffuser, "n_steps", "?")
            lines.append(
                f"[Stage] train timestep : from dataset, range 1..{n_steps} (no override)"
            )

        # 2. Timestep bucketing.  Keyed on the CALL SITE in run(), not on whether
        #    _bucket_timestep exists: the method body survives being commented out
        #    at the call site, and matching the body would report a dead knob as
        #    active.  The representatives still come from the body -- that is the
        #    anti-drift part -- but only once the call is confirmed live.
        if re.search(r"_bucket_timestep\s*\(", run_src):
            bucket_src = _live(_src(getattr(type(self.diffuser), "_bucket_timestep", None)))
            reps = re.findall(r"^\s*return\s+(\d+)\s*$", bucket_src, flags=re.MULTILINE)
            lines.append(
                f"[Stage] t bucketing    : ACTIVE -- every t collapses onto "
                f"{{{', '.join(reps) if reps else '?'}}} "
                f"({len(reps)} distinct sigma levels reach the net)"
            )
        else:
            lines.append(
                "[Stage] t bucketing    : off -- full t range reaches the net"
            )

        # 3. Noise seeding.  A global manual_seed inside run() also pins dropout,
        #    since dropout draws from the same global RNG.
        seeds_priv = re.findall(r"generator\.manual_seed\(\s*(\d+)\s*\)", run_src)
        seeds_glob = re.findall(r"^\s*torch\.manual_seed\(\s*(\d+)\s*\)", run_src,
                                flags=re.MULTILINE)
        if seeds_priv or seeds_glob:
            note = (
                f"rota private generator={seeds_priv[-1] if seeds_priv else 'free'}, "
                f"global RNG={seeds_glob[-1] if seeds_glob else 'free'}"
            )
            lines.append(f"[Stage] noise seeding  : PINNED per run() call -- {note}")
            if seeds_glob:
                lines.append(
                    "[Stage]                  WARNING global manual_seed re-seeds every "
                    "call, so DROPOUT masks repeat identically each step (regularisation "
                    "effectively disabled)"
                )
        else:
            lines.append("[Stage] noise seeding  : free (random every call)")

        # 4. Cheap-layer validation timesteps.  Read straight off the attribute:
        #    unlike the old inline `batch["idx_step"] = int(75)` literals buried
        #    in validation_step, a class constant IS the live value, so there is
        #    nothing to drift away from.
        val_ts = [str(int(t)) for t in self.CHEAP_VAL_TIMESTEPS]
        if val_ts:
            eff = f" -- but overridden to {pinned[-1]} by run()" if pinned else ""
            lines.append(
                f"[Stage] val timesteps  : {', '.join(val_ts)}"
                f" (cheap layer, each logged separately as val_t<N>/, "
                f"fixed seed={self.cheap_eval_seed}){eff}"
            )
        else:
            lines.append("[Stage] val timesteps  : from dataset (not pinned)")

        # 4b. Precise layer: reverse sampling.  Note it is NOT affected by the
        #     run() override above -- it calls _run_impl directly.
        if self.precise_eval_every_n_val > 0:
            n_sched = len(self.diffuser.build_reverse_schedule(self.precise_eval_steps)) - 1
            lines.append(
                f"[Stage] precise layer  : ON -- every {self.precise_eval_every_n_val} "
                f"val run(s), reverse sampling with {n_sched} network evaluations "
                f"(steps={self.precise_eval_steps or 'full'}), seed={self.precise_eval_seed}, "
                f"metrics reported as means under 'val/precise_*'"
            )
            lines.append(
                "[Stage]                  start = TRUE MARGINAL "
                "(sqrt(sigma_T^2 + sigma_data^2)), rotation = Haar; "
                "run()'s pinned t does NOT apply here"
            )
        else:
            lines.append(
                "[Stage] precise layer  : OFF (precise_eval_every_n_val=0) -- "
                "no reverse sampling, so val/precise_tm_score is never logged"
            )

        # 5. Knobs that come from config rather than source text.
        # Lightning's `trainer` is a property that RAISES when no Trainer is
        # attached, so a plain getattr(..., None) does not make this safe.
        try:
            accum = self.trainer.accumulate_grad_batches
        except Exception:
            accum = "?"
        lines.append(
            f"[Stage] ema            : "
            f"{'on, decay=' + str(self.ema.decay) + ' (val runs on EMA weights)' if self.ema is not None else 'OFF (no ema_decay passed)'}"
        )
        lines.append(
            f"[Stage] accum / lr     : accumulate_grad_batches={accum}, "
            f"lr={self.optimizer_cfg.lr}"
        )
        lines.append("[Stage] " + "-" * 52)

        for ln in lines:
            print(ln, flush=True)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    # Cheap validation layer: the timesteps at which every checkpoint is scored.
    # Frozen deliberately -- a moving t makes two checkpoints incomparable.  Each
    # gets its own log prefix rather than being averaged into one "val/" number,
    # because c_out spans two orders of magnitude across this range, so an average
    # over them is dominated by whichever t happens to be noisiest.
    CHEAP_VAL_TIMESTEPS = (25, 75, 125, 175)

    def _fixed_cheap_seed(self, prot_id: str, timestep: int) -> int:
        payload = f"{self.cheap_eval_seed}:{prot_id}:{int(timestep)}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "little") & ((1 << 63) - 1)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Cheap layer: fixed-noise single-step losses plus existing loop AAR/RMSD.
        loss = None
        for t in self.CHEAP_VAL_TIMESTEPS:
            batch["idx_step"] = int(t)
            loss = self._shared_step(
                batch,
                stage="val",
                log_prefix=f"val_t{t}",
                compute_metrics=False,
                compute_cheap_metrics=True,
                fixed_noise_seed=self._fixed_cheap_seed(batch.get("prot_id", ""), t),
            )
        # Keep the batch for the precise layer, which runs in
        # on_validation_epoch_end (so that it lands before ema.restore()).
        if self._precise_eval_due():
            self._pending_precise_batches.append(batch)
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        group = batch.get("test_group")
        stage = "test"
        if isinstance(group, str) and group:
            stage = f"test_{self._safe_log_name(group)}"
        return self._shared_step(batch, stage=stage)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep PLM featurizer frozen in eval mode while DesignModel trains.
        self.plm_featurizer.eval()
        return self

    def configure_optimizers(self):
        cfg = self.optimizer_cfg

        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for optimizer setup.")

        if cfg.name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=cfg.lr,
                betas=cfg.betas,
                eps=cfg.eps,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.name}")

        if not self.scheduler_cfg:
            return optimizer

        sched_name = self.scheduler_cfg.get("name", "cosine").lower()
        warmup_steps = int(self.scheduler_cfg.get("warmup_steps", 0))

        # 1. 定义主调度器
        if sched_name == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.scheduler_cfg.get("t_max", 1000)),
                eta_min=float(self.scheduler_cfg.get("eta_min", 1e-6)),
            )
        elif sched_name == "multistep":
            main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=list(self.scheduler_cfg.get("milestones", [1000, 2000])),
                gamma=float(self.scheduler_cfg.get("gamma", 0.1)),
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_name}")

        # 2. 如果存在 Warm-up，则组合调度器
        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.01, # 初始学习率为 base_lr * 0.01
                total_iters=warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, 
                schedulers=[warmup_scheduler, main_scheduler], 
                milestones=[warmup_steps]
            )
        else:
            scheduler = main_scheduler

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm) -> None:
        clip_val = self.grad_clip_val if self.grad_clip_val is not None else gradient_clip_val
        if clip_val is None:
            return
        self.clip_gradients(optimizer, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")


    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """Catch backward OOM and skip optimizer step to keep long runs alive."""
        try:
            loss.backward(*args, **kwargs)
        except RuntimeError as exc:
            if not self._is_oom_error(exc):
                raise
            self._skip_optimizer_step_due_to_oom = True
            self.log("train/skip_backward_oom", torch.tensor(1.0, device=self.device), prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            print(f"[IgGMLightningModule] backward OOM detected, skip optimizer step: {exc}")
            for p in self.model.parameters():
                p.grad = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_closure) -> None:
        # DDP-safe: if any rank hit backward OOM, all ranks skip this optimizer step.
        skip_step = self._ddp_any_true(self._skip_optimizer_step_due_to_oom)
        self._skip_optimizer_step_due_to_oom = False
        if skip_step:
            optimizer_closure()
            optimizer.zero_grad(set_to_none=True)
            self.log("train/skip_optim_step_oom", torch.tensor(1.0, device=self.device), prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            return
        optimizer.step(closure=optimizer_closure)


    def on_before_zero_grad(self, optimizer) -> None:
        # EMA is updated per OPTIMIZER step, not per batch.  on_train_batch_end
        # fires once per microbatch, so under accumulate_grad_batches=N it would
        # tick N times per update and shrink the effective EMA horizon to 1/N.
        if self.ema is not None:
            self.ema.update(self.model)

    def _is_sanity_checking(self) -> bool:
        # Lightning's `trainer` is a property that RAISES when unattached, so the
        # access itself needs guarding, not just the attribute.
        try:
            return bool(self.trainer.sanity_checking)
        except Exception:
            return False

    def on_validation_start(self) -> None:
        # Do NOT count the sanity-check pass.  It runs the validation hooks before
        # training has produced anything, and _precise_eval_due skips it -- so
        # counting it would burn slot 1 and shift the whole cadence by one, i.e.
        # the first REAL validation would silently not run the precise layer.
        if not self._is_sanity_checking():
            self._val_run_counter += 1
        self._pending_precise_batches = []
        # Validate with the EMA weights: they only affect the eval path, never
        # the training gradients, so this does not perturb training dynamics.
        if self.ema is not None:
            self.ema.copy_to(self.model)

    def on_validation_epoch_end(self) -> None:
        # The precise layer must run BEFORE ema.restore(), so that it scores the
        # same EMA weights the cheap layer just did.
        self._run_precise_eval()
        # Logged unconditionally (carrying forward on non-precise runs) so the
        # precise ModelCheckpoint's monitor is never missing -- see the docstring.
        if self.precise_eval_every_n_val > 0 and not self._is_sanity_checking():
            self._log_precise_metrics()

        # Restore here rather than in on_validation_end: Lightning runs CALLBACK
        # on_validation_end (where ModelCheckpoint writes the file) BEFORE the
        # LightningModule hook of the same name, so restoring there would save
        # the EMA weights into `state_dict` and make last.ckpt resume training
        # from EMA weights.  The monitored metric still comes from the EMA pass;
        # the EMA weights themselves are persisted separately, see
        # on_save_checkpoint.
        if self.ema is not None:
            self.ema.restore(self.model)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        shadow = checkpoint.get("ema_shadow")
        if self.ema is not None and shadow is not None:
            self.ema.shadow = {
                k: v.to(self.ema.shadow[k].device)
                for k, v in shadow.items()
                if k in self.ema.shadow
            }

    # def on_after_backward(self):
    #     fr = self.model.net["af2_smod"].net["fr_branch"]
    #     tw = fr.trsl_head[-1].weight.grad
    #     qw = fr.rota_head[-1].weight.grad
    #     print("trsl_head.weight.grad:", None if tw is None else tw.norm().item())
    #     print("rota_head.weight.grad:", None if qw is None else qw.norm().item())
