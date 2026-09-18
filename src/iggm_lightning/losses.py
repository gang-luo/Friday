"""
Losses for the Lightning training path.

The CDR coordinate loss supervises the model's predicted clean loop-local
coordinates with the diffuser-provided clean_loop_local_coords.  These local
coordinates are invariant to a shared rigid transform of the antibody and its
anchors, so they are the canonical training target for the local CDR denoiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional,Sequence
import torch
import torch.nn.functional as F

from IgGM.protein.prot_constants import (
    RESD_MAP_1TO3,
    RESD_NAMES_1C,
    SIDECHAIN_BONDS_PER_RESD,
    restype_atom14_to_atom37,
    restype_name_to_atom14_names,
)
from openfold.utils.loss import find_structural_violations, violation_loss
from IgGM.utils.diff_util import so3_log_vector


# @dataclass
# class IgGMLossConfig:
#     backbone_weight: float = 1.0
#     cdr_all_atom_weight: float = 10.0
#     smooth_lddt_weight: float = 1.0
#     bond_weight: float = 1.0
#     vio_weight: float = 0.02
#     # A3: min-SNR loss weighting (Hang et al. 2023). Off by default; enable when
#     # scaling to many samples to balance gradients across noise levels.
#     use_snr_weight: bool = False
#     snr_gamma: float = 0.5 # 5.0

@dataclass
class IgGMLossConfig:
    backbone_weight: float = 1.0
    # AB2: lowered 10.0 -> 1.0 for the first run with loss_cdr enabled, so the
    # CDR term cannot swamp the FR gradients while we check whether FR degrades.
    # Old: cdr_all_atom_weight: float = 10.0
    cdr_all_atom_weight: float = 1.0
    smooth_lddt_weight: float = 1.0
    bond_weight: float = 1.0
    # SUPERSEDED, NOT DEAD CODE -- these four have NO `self.cfg.<name>` reference
    # anywhere in this file, so changing them has no effect.  They belonged to the
    # earlier scheme that supervised the type channel directly (marker topology /
    # marker count / marker AAR / loop closure).  That job is now done by the
    # virtual-atom component of loss_cdr plus the seq-head CE (loss_seq), which
    # together reached aar_cdr 1.000 -- so there is no reason to revive them.
    # Kept as a record that the direct-marker-supervision route was tried and
    # replaced, the same way _cdr_smooth_lddt_loss keeps its verdict in-place.
    closure_weight: float = 1.0
    marker_topology_weight: float = 0.25
    marker_count_weight: float = 1.0
    marker_aar_weight: float = 0.5
    vio_weight: float = 0.02

    fr_global_aux_weight: float = 0.1
    # Scheme B: sequence-head cross-entropy (co-design). 0.0 -> inert.
    # KEEP: this head is an auxiliary supervision channel only -- residue type is
    # decoded from atom14 virtual-atom geometry, never from these logits.  Do not
    # delete it as dead code; the call site is commented out on purpose while the
    # CDR losses are being brought up one at a time.
    seq_head_weight: float = 0.5
    marker_distance_threshold: float = 0.5
    marker_distance_temperature: float = 0.08
    marker_assignment_temperature: float = 0.08
    marker_aar_temperature: float = 0.05
    marker_count_normalizer: float = 10.0

    use_snr_weight: bool = False
    snr_gamma: float = 0.5

class IgGMPaperLoss:
    """Compute aligned backbone/CDR/vio loss with per-layer CDR supervision."""

    def __init__(self, cfg: IgGMLossConfig | None = None) -> None:
        self.cfg = cfg or IgGMLossConfig()
        self._aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        (
            self._nominal_atom14_mask_cpu,
            self._sidechain_edge_indices_cpu,
            self._sidechain_edge_mask_cpu,
        ) = self._build_atom14_loss_tables()
        self._atom14_table_cache = {}
        self.idx_save = 0

    def __call__(self, inputs: Dict, outputs: Dict) -> Dict:
        return self._aligned_backbone_cdr_vio_loss(inputs, outputs)

    # ----------------------------------------------------------------
    # 静态工具（不变）
    # ----------------------------------------------------------------
    @staticmethod
    def _ensure_batched(t, ndim_no_batch):
        return t.unsqueeze(0) if t.ndim == ndim_no_batch else t

    @staticmethod
    def _build_atom14_loss_tables():
        max_sidechain_edges = max(
            len(SIDECHAIN_BONDS_PER_RESD[RESD_MAP_1TO3[aa]])
            for aa in RESD_NAMES_1C
        )
        nominal_mask = torch.zeros((len(RESD_NAMES_1C), 14), dtype=torch.bool)
        edge_indices = torch.zeros(
            (len(RESD_NAMES_1C), max_sidechain_edges, 2), dtype=torch.long
        )
        edge_mask = torch.zeros(
            (len(RESD_NAMES_1C), max_sidechain_edges), dtype=torch.bool
        )

        for type_idx, aa in enumerate(RESD_NAMES_1C):
            resname = RESD_MAP_1TO3[aa]
            atom_names = restype_name_to_atom14_names[resname]
            atom_to_idx = {
                atom_name: atom_idx
                for atom_idx, atom_name in enumerate(atom_names)
                if atom_name
            }
            nominal_mask[type_idx] = torch.tensor(
                [bool(atom_name) for atom_name in atom_names], dtype=torch.bool
            )

            seen_edges = set()
            for edge_idx, (atom1_name, atom2_name) in enumerate(
                SIDECHAIN_BONDS_PER_RESD[resname]
            ):
                if atom1_name not in atom_to_idx or atom2_name not in atom_to_idx:
                    raise ValueError(
                        f"Invalid sidechain bond for {resname}: "
                        f"{atom1_name}-{atom2_name}"
                    )
                atom1_idx = atom_to_idx[atom1_name]
                atom2_idx = atom_to_idx[atom2_name]
                edge_key = tuple(sorted((atom1_idx, atom2_idx)))
                if edge_key in seen_edges:
                    raise ValueError(
                        f"Duplicate sidechain bond for {resname}: "
                        f"{atom1_name}-{atom2_name}"
                    )
                seen_edges.add(edge_key)
                edge_indices[type_idx, edge_idx] = torch.tensor(
                    [atom1_idx, atom2_idx], dtype=torch.long
                )
                edge_mask[type_idx, edge_idx] = True

        return nominal_mask, edge_indices, edge_mask

    def _atom14_loss_tables(self, device: torch.device):
        cache_key = (device.type, device.index)
        tables = self._atom14_table_cache.get(cache_key)
        if tables is None:
            tables = (
                self._nominal_atom14_mask_cpu.to(device=device),
                self._sidechain_edge_indices_cpu.to(device=device),
                self._sidechain_edge_mask_cpu.to(device=device),
            )
            self._atom14_table_cache[cache_key] = tables
        return tables

    @staticmethod
    def _masked_per_residue_mean(values: torch.Tensor, mask: torch.Tensor):
        mask_float = mask.to(dtype=values.dtype)
        atom_or_edge_count = mask_float.sum(dim=-1)
        per_residue = (values * mask_float).sum(dim=-1) / atom_or_edge_count.clamp_min(1.0)
        residue_active = atom_or_edge_count > 0
        residue_active_float = residue_active.to(dtype=values.dtype)
        per_sample = (
            (per_residue * residue_active_float).flatten(1).sum(dim=-1)
            / residue_active_float.flatten(1).sum(dim=-1).clamp_min(1.0)
        )
        sample_active = residue_active.flatten(1).any(dim=-1)
        return per_sample, sample_active

    @staticmethod
    def _aggregate_active_groups(group_values, group_active):
        values = torch.stack(group_values, dim=-1)
        active = torch.stack(group_active, dim=-1)
        active_float = active.to(dtype=values.dtype)
        total_per_sample = (
            (values * active_float).sum(dim=-1)
            / active_float.sum(dim=-1).clamp_min(1.0)
        )
        component_losses = (
            (values * active_float).sum(dim=0)
            / active_float.sum(dim=0).clamp_min(1.0)
        )
        return total_per_sample.mean(), tuple(component_losses.unbind(dim=0))

    def _log_active_loss_terms(self, local_vars):
        """AC1: print, once, which loss terms actually reach backward().

        Why this exists: on 2026-08-31 a 1000-step run was wasted because
        loss_cdr was computed and logged to wandb but never added to `total`.
        Every surface signal said "it is training" -- only reading the commented
        lines of the `total = (...)` block revealed otherwise.

        Rather than duplicating the term list (which would drift), this parses
        the `total = (...)` block out of this class's own source, so the banner
        is derived from the exact text you edit when toggling a term.
        """
        if getattr(self, "_active_terms_logged", False):
            return
        self._active_terms_logged = True

        import inspect
        import re

        try:
            src = inspect.getsource(type(self)).splitlines()
        except OSError:
            return

        start = next(
            (i for i, ln in enumerate(src) if re.match(r"\s*total\s*=\s*\($", ln)),
            None,
        )
        if start is None:
            return

        active, inactive = [], []
        for ln in src[start + 1:]:
            if re.match(r"\s*\)\s*$", ln):
                break
            stripped = ln.lstrip()
            commented = stripped.startswith("#")
            # Only count lines that are actual summands -- a disabled term looks
            # like "# + self.cfg.x_weight * loss_x".  Prose comments that merely
            # mention a loss name (e.g. "loss_cdr enabled ...") must not be
            # mistaken for a toggled-off term.
            body = stripped.lstrip("#").strip() if commented else stripped
            if not (body.startswith("+") or body.startswith("self.cfg")):
                continue
            m = re.search(r"(loss_[A-Za-z0-9_]+)", body)
            if not m:
                continue
            (inactive if commented else active).append(m.group(1))

        def fmt(names):
            out = []
            for n in names:
                v = local_vars.get(n)
                try:
                    out.append(f"{n}={float(v):.4g}")
                except (TypeError, ValueError):
                    out.append(n)
            return ", ".join(out) if out else "(none)"

        print(f"[Loss] IN total (has gradient): {fmt(active)}", flush=True)
        print(f"[Loss] NOT in total (logged only): {', '.join(inactive) or '(none)'}",
              flush=True)

    @staticmethod
    def _rotation_diagnostics(outputs, target):
        pred = outputs["3d"]["rota_vec_norm"][-1].float()
        return {
            "pred_energy": pred.square().sum(-1).mean().detach(),
            "target_energy": target.square().sum(-1).mean().detach(),
            "dot": (pred * target).sum(-1).mean().detach(),
        }

    @staticmethod
    def _so3_angle(rotation: torch.Tensor) -> torch.Tensor:
        """Rotation angle in radians, from the trace. Continuous on [0, pi]."""
        cos = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
        return torch.arccos(cos)

    @classmethod
    def _rotation_absorption(
        cls,
        pred_rota_vec_norm: torch.Tensor,
        target_delta_rota: torch.Tensor,
        rota_rms: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """AB1: geometric absorption, comparable across noise buckets.

        Returns the two angles whose ratio defines it:
            absorption = 1 - theta(R_pred^T R_tgt) / theta(R_tgt)
        i.e. the fraction of the required correction the model actually applied.

        Unlike the loss value, this is a pure angle ratio: it is unaffected by
        the loss parameterisation (see Y1, which changed the numeric range and
        made historical loss curves incomparable) and by which noise bucket is
        being trained.  It is therefore the only rotation metric that can be
        compared across experiments.
        """
        vec = pred_rota_vec_norm.float() * rota_rms.float().reshape(-1, 1).clamp_min(1e-6)
        pred_delta = cls._so3_exp_map_local(vec)
        residual = pred_delta.transpose(-1, -2) @ target_delta_rota.float()
        return {
            "target_angle": cls._so3_angle(target_delta_rota.float()).mean().detach(),
            "residual_angle": cls._so3_angle(residual).mean().detach(),
        }

    # ------------------------------------------------------------------
    # Y1: Frobenius-distance rotation loss (alternative to log-map MSE)
    # ------------------------------------------------------------------
    @staticmethod
    def _so3_exp_map_local(vector: torch.Tensor) -> torch.Tensor:
        """Rodrigues exp-map, identical to FRBranch._so3_exp_map.

        Duplicated here (rather than imported) so the loss cannot silently
        drift from the reconstruction path in fr_cdr_blocks.py; if you change
        one, change both.
        """
        theta_sq = vector.square().sum(-1, keepdim=True)
        theta = theta_sq.clamp_min(1e-12).sqrt()
        theta_safe = theta.clamp_min(1e-4)

        coef_a = torch.sin(theta_safe) / theta_safe
        coef_b = (1.0 - torch.cos(theta_safe)) / theta_safe.square()
        coef_a = torch.where(
            theta_sq < 1e-8, 1.0 - theta_sq / 6.0 + theta_sq.square() / 120.0, coef_a
        )
        coef_b = torch.where(
            theta_sq < 1e-8, 0.5 - theta_sq / 24.0 + theta_sq.square() / 720.0, coef_b
        )

        x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
        zero = torch.zeros_like(x)
        skew = torch.stack(
            [
                torch.stack([zero, -z, y], dim=-1),
                torch.stack([z, zero, -x], dim=-1),
                torch.stack([-y, x, zero], dim=-1),
            ],
            dim=-2,
        )
        eye = torch.eye(3, device=vector.device, dtype=vector.dtype)
        eye = eye.view(*((1,) * (vector.ndim - 1)), 3, 3)
        return eye + coef_a.unsqueeze(-1) * skew + coef_b.unsqueeze(-1) * (skew @ skew)

    @classmethod
    def _rota_frobenius_loss(
        cls,
        pred_rota_vec_norm: torch.Tensor,
        target_delta_rota: torch.Tensor,
        rota_rms: torch.Tensor,
    ) -> torch.Tensor:
        """||R_pred - R_tgt||_F^2 in the delta-rotation space.

        Compared with MSE on the normalised log-map, this metric is
          * continuous everywhere on SO(3) (no 2-pi wrap near theta=pi),
          * free of the 1/rota_rms amplification that makes the objective
            inconsistent across noise buckets,
          * equal to 4*(1 - cos theta) = 8 sin^2(theta/2), i.e. a monotone
            function of the geodesic distance that saturates instead of
            diverging.

        The head parameterisation is unchanged: it still emits a normalised
        rotation vector which is exponentiated with rota_rms as the scale.
        """
        vec = pred_rota_vec_norm.float() * rota_rms.reshape(-1, 1).clamp_min(1e-6)
        pred_delta = cls._so3_exp_map_local(vec)
        diff = pred_delta - target_delta_rota.float()
        return diff.pow(2).flatten(1).sum(-1).mean()

    # @staticmethod
    # def _so3_log_vector(rotation: torch.Tensor) -> torch.Tensor:
    #     rotation = rotation.float()
    #     skew = 0.5 * torch.stack(
    #         [
    #             rotation[..., 2, 1] - rotation[..., 1, 2],
    #             rotation[..., 0, 2] - rotation[..., 2, 0],
    #             rotation[..., 1, 0] - rotation[..., 0, 1],
    #         ],
    #         dim=-1,
    #     )
    #     sin_angle = torch.linalg.norm(skew, dim=-1)
    #     cos_angle = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    #     angle = torch.atan2(sin_angle, cos_angle)
    #     rotvec = skew * (angle / sin_angle.clamp_min(1e-7)).unsqueeze(-1)
    #     rotvec = torch.where((angle < 1e-5).unsqueeze(-1), skew, rotvec)

    #     near_pi = (torch.pi - angle).abs() < 1e-4
    #     if near_pi.any():
    #         sym = 0.5 * (
    #             rotation[near_pi]
    #             + torch.eye(3, device=rotation.device, dtype=torch.float32)
    #         )
    #         _, eigenvectors = torch.linalg.eigh(sym)
    #         axes = eigenvectors[..., -1]
    #         max_indices = axes.abs().argmax(dim=-1, keepdim=True)
    #         signs = torch.gather(axes, -1, max_indices).sign()
    #         signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    #         rotvec[near_pi] = axes * signs * angle[near_pi].unsqueeze(-1)
    #     return rotvec


    def _fr_residual_loss(self, inputs: Dict, outputs: Dict):
        meta = inputs["anchor_frame_meta"]
        trsl_predictions = outputs["3d"]["trsl_residual"]
        rota_predictions = outputs["3d"]["rota_vec_norm"]
        if not trsl_predictions or len(trsl_predictions) != len(rota_predictions):
            raise ValueError("FR residual layer outputs are missing or inconsistent")

        batch_size = trsl_predictions[-1].shape[0]
        device = trsl_predictions[-1].device
        with torch.autocast(device_type=device.type, enabled=False):
            rota_xt = meta["rota_xt"].to(device=device, dtype=torch.float32)
            rota_orig = meta["rota_orig"].to(device=device, dtype=torch.float32)
            trsl_xt = meta["trsl_xt_physical"].to(device=device, dtype=torch.float32).reshape(-1, 3)
            trsl_orig = meta["trsl_orig"].to(device=device, dtype=torch.float32).reshape(-1, 3)
            antigen_com = meta["antigen_com"].to(device=device, dtype=torch.float32).reshape(-1, 3)

            if rota_xt.ndim == 2:
                rota_xt = rota_xt.unsqueeze(0)
            if rota_orig.ndim == 2:
                rota_orig = rota_orig.unsqueeze(0)
            if rota_xt.shape[0] == 1 and batch_size > 1:
                rota_xt = rota_xt.expand(batch_size, -1, -1)
                rota_orig = rota_orig.expand(batch_size, -1, -1)
                trsl_xt = trsl_xt.expand(batch_size, -1)
                trsl_orig = trsl_orig.expand(batch_size, -1)
                antigen_com = antigen_com.expand(batch_size, -1)

            c_skip = meta["fr_c_skip"].to(device=device, dtype=torch.float32).reshape(-1)
            c_out = meta["fr_c_out"].to(device=device, dtype=torch.float32).reshape(-1)
            rota_rms = meta["fr_rota_rms"].to(device=device, dtype=torch.float32).reshape(-1)
            if c_skip.numel() == 1:
                c_skip = c_skip.expand(batch_size)
                c_out = c_out.expand(batch_size)
                rota_rms = rota_rms.expand(batch_size)

            trsl_xt_body = torch.matmul((trsl_xt - antigen_com).unsqueeze(1), rota_xt).squeeze(1)
            trsl_orig_body = torch.matmul((trsl_orig - antigen_com).unsqueeze(1), rota_xt).squeeze(1)
            target_trsl_residual = (
                trsl_orig_body - c_skip.unsqueeze(-1) * trsl_xt_body
            ) / c_out.unsqueeze(-1).clamp_min(1e-6)

            target_delta_rota = rota_xt.transpose(-1, -2) @ rota_orig
            target_rota_vec_norm = so3_log_vector(target_delta_rota)
            target_rota_vec_norm = target_rota_vec_norm / rota_rms.unsqueeze(-1).clamp_min(1e-6)

            loss_trsl_residual = target_trsl_residual.new_tensor(0.0)
            loss_rota_residual = target_rota_vec_norm.new_tensor(0.0)
            weight_sum = 0.0
            for layer_idx, (pred_trsl, pred_rota) in enumerate(
                zip(trsl_predictions, rota_predictions)
            ):
                layer_weight = float(layer_idx + 1)
                weight_sum += layer_weight
                loss_trsl_residual = loss_trsl_residual + layer_weight * F.mse_loss(
                    pred_trsl.float(), target_trsl_residual
                )
                # --- Y1: old path, MSE on the normalised log-map ---
                # Kept for A/B comparison.  Pathology: the 1/rota_rms
                # normalisation makes the objective inconsistent across noise
                # buckets, and the log-map is discontinuous near theta=pi.
                # loss_rota_residual = loss_rota_residual + layer_weight * F.mse_loss(
                #     pred_rota.float(), target_rota_vec_norm
                # )
                # --- Y1: new path, Frobenius distance on SO(3) ---
                loss_rota_residual = loss_rota_residual + layer_weight * (
                    self._rota_frobenius_loss(
                        pred_rota, target_delta_rota, rota_rms
                    )
                )
            loss_trsl_residual = loss_trsl_residual / weight_sum
            loss_rota_residual = loss_rota_residual / weight_sum

        rotation_diag = self._rotation_diagnostics(
            outputs=outputs,
            target=target_rota_vec_norm,
        )
        # AB1: absorption on the last layer only (that is the layer whose
        # prediction is actually used downstream).
        rotation_diag.update(
            self._rotation_absorption(
                rota_predictions[-1], target_delta_rota, rota_rms
            )
        )
        return loss_trsl_residual, loss_rota_residual, rotation_diag

    @staticmethod
    def _normalize_res_mask(mask, batch_size, seq_len):
        mask = mask.to(torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        if mask.shape != (batch_size, seq_len):
            raise ValueError(f"Unexpected residue mask shape: {tuple(mask.shape)}")
        return mask

    @staticmethod
    def _kabsch_transform(src, tgt, valid):
        if int(valid.sum().item()) < 3:
            return torch.eye(3, device=src.device, dtype=src.dtype), torch.zeros(3, device=src.device, dtype=src.dtype)
        src_sel = src[valid].float()
        tgt_sel = tgt[valid].float()
        src_mean, tgt_mean = src_sel.mean(0), tgt_sel.mean(0)
        src0, tgt0 = src_sel - src_mean, tgt_sel - tgt_mean
        cov = src0.t() @ tgt0
        u, _, vh = torch.linalg.svd(cov.float())
        rot = vh.t() @ u.t()
        if torch.det(rot.float()) < 0:
            vh[-1] *= -1
            rot = vh.t() @ u.t()
        tr = tgt_mean - src_mean @ rot.t()
        return rot.to(dtype=src.dtype), tr.to(dtype=src.dtype)

    def _align_pred_to_target(self, pred, tgt, atom_mask, align_res_mask, align_atom_idx, asym_id, align_mode="complex"):
        aligned = pred.clone()
        bsz = pred.shape[0]
        for b in range(bsz):
            base_mask = align_res_mask[b]
            chain_mask = torch.ones_like(base_mask, dtype=torch.bool)
            if asym_id is not None and asym_id.numel() > 0:
                if align_mode == "antigen":
                    chain_mask = (asym_id[b] == 0)
                elif align_mode == "antibody":
                    chain_mask = (asym_id[b] != 0)
                elif align_mode.startswith("chain_"):
                    try:
                        chain_mask = (asym_id[b] == int(align_mode.split("_")[1]))
                    except (IndexError, ValueError):
                        pass
                effective_mask = base_mask & chain_mask
            else:
                effective_mask = base_mask
            align_atoms = atom_mask[b, :, align_atom_idx].all(dim=-1) & effective_mask
            src = pred[b, :, align_atom_idx].reshape(-1, 3)
            dst = tgt[b, :, align_atom_idx].reshape(-1, 3)
            val = align_atoms[:, None].expand(-1, len(align_atom_idx)).reshape(-1)
            rot, tr = self._kabsch_transform(src, dst, val)
            aligned[b] = torch.matmul(pred[b], rot.t()) + tr.view(1, 1, 3)
        return aligned

    def _seq_to_aatype(self, seq_obj, bsz, seq_len, device):
        aa_to_idx = {aa: i for i, aa in enumerate(RESD_NAMES_1C)}
        seqs = [seq_obj] if isinstance(seq_obj, str) else list(seq_obj)
        if len(seqs) == 1 and bsz > 1:
            seqs = seqs * bsz
        out = torch.full((bsz, seq_len), 20, dtype=torch.long, device=device)
        for b, seq in enumerate(seqs):
            seq = "".join(str(x) for x in seq) if isinstance(seq, (list, tuple)) else str(seq)
            n = min(len(seq), seq_len)
            if n > 0:
                out[b, :n] = torch.tensor([aa_to_idx.get(ch, 20) for ch in seq[:n]], device=device, dtype=torch.long)
        return out

    @staticmethod
    def _gather_full_tensor_to_loops(
        full_tensor: torch.Tensor,
        loop_global_res_indices: torch.Tensor,
        loop_valid_res_mask: torch.Tensor,
        fill_value: float | int,
    ) -> torch.Tensor:
        """Utility to extract loop-specific features from a full-sequence tensor."""
        if loop_global_res_indices.ndim == 2: loop_global_res_indices = loop_global_res_indices.unsqueeze(0)
        if loop_valid_res_mask.ndim == 2: loop_valid_res_mask = loop_valid_res_mask.unsqueeze(0)
        # Normalize to [B, L, *tail]:
        #   per-residue scalar label [L] -> [1, L]  (empty tail)
        #   unbatched feature tensor  [L, feat] -> [1, L, feat]
        if full_tensor.ndim == 1:
            full_tensor = full_tensor.unsqueeze(0)          # [1, L]
        elif full_tensor.ndim == 2 and full_tensor.shape[0] != loop_global_res_indices.shape[0]:
            full_tensor = full_tensor.unsqueeze(0)          # [1, L, feat]

        batch_size, n_loop, lmax = loop_global_res_indices.shape
        if full_tensor.shape[0] == 1 and batch_size > 1:
            full_tensor = full_tensor.expand(batch_size, *full_tensor.shape[1:])
        if full_tensor.shape[0] != batch_size:
            raise ValueError("Full tensor batch size does not match loop indices")

        seq_len = full_tensor.shape[1]
        tail_shape = full_tensor.shape[2:]

        # Expand source for gathering
        source = full_tensor.unsqueeze(1).expand(batch_size, n_loop, seq_len, *tail_shape)
        safe_indices = loop_global_res_indices.clamp(min=0, max=seq_len - 1)

        index_shape = (batch_size, n_loop, lmax, *([1] * len(tail_shape)))
        gather_indices = safe_indices.view(index_shape).expand(batch_size, n_loop, lmax, *tail_shape)

        gathered = torch.gather(source, dim=2, index=gather_indices)

        valid_shape = (batch_size, n_loop, lmax, *([1] * len(tail_shape)))
        valid = loop_valid_res_mask.view(valid_shape)
        fill = torch.as_tensor(fill_value, device=gathered.device, dtype=gathered.dtype)

        return torch.where(valid, gathered, fill)

    @staticmethod
    def _gather_edge_endpoints(coords, edge_indices):
        n_edges = edge_indices.shape[-2]
        source = coords.unsqueeze(3).expand(-1, -1, -1, n_edges, -1, -1)
        gather_indices = edge_indices.unsqueeze(-1).expand(-1, -1, -1, -1, -1, 3)
        return torch.gather(source, dim=4, index=gather_indices)
    
    # ----------------------------------------------------------------
    # smooth lDDT
    # ----------------------------------------------------------------
    def _cdr_smooth_lddt_loss(
        self,
        pred_coords,
        true_coords,
        atom14_mask,
        cdr_mask,
        cutoff=15.0,
        sharpness=10.0,
    ):
        """Smooth lDDT over CDR atoms, in GLOBAL coordinates.

        Two facts about this term that are easy to get wrong:

        1. It does NOT backpropagate into the FR pose.  fr_cdr_blocks.py does
           `frame_source = fr_coords.detach()`, so the anchor frames used to lift
           the loop prediction to global coordinates are detached; _merge_fr_cdr
           only overwrites CDR rows.  All gradient lands on the CDR head.

        2. Rewriting it in loop-local coordinates would make it redundant.
           lDDT depends only on interatomic distances, which are invariant to
           the rigid anchor transform, so the INTRA-loop contribution is
           numerically identical either way.  Its unique signal is the
           CROSS-loop relative geometry, which the per-loop local MSE in
           _cdr_grouped_atom_mse cannot see.  Note the scope: `valid_mask` below
           is `cdr_mask & atom14_mask`, so ONLY CDR atoms enter the pair set --
           antigen atoms do not, and there is no loop-to-antigen term here.

        Caveat for scheduling: `true_coords` are clean global coordinates while
        the predicted anchor frames come from the current layer's predicted FR.
        Any FR pose error therefore injects a systematic bias into the
        cross-loop distances that the CDR head can only absorb as deformation.
        Enable this only once FR is stable.

        MEASURED VERDICT (2026-09-01), why this term is left out of `total`.

        Its unique signal turned out to be already supervised.  Cross-loop CA
        hit@1A (see/probe_af1_cross_loop.py) reaches 1.000 by the last dump in
        BOTH runs that had this term off -- each loop's shape is pinned by its
        own local MSE and the loops' relative placement follows from the FR
        anchors, which loss_backbone already supervises.  Enabling it only made
        that number converge ~2x sooner; the endpoint was identical.

        The cost is the type channel.  Aligned by dump index (dumps land on a
        fixed epoch cadence, so row i is the same epoch across runs):

            dump 9        anch_d   frac decode   aar_cdr
            cdr_on          0.81         0.802     0.468
            bond+norm       0.83         0.780     0.617
            +smooth_lddt    2.46         0.160     0.234

        anch_d is |marker - its own N/O anchor| in the prediction; the ground
        truth is exactly 0 and decode_threshold is 1.0 A, so that column IS the
        type channel.  Backbone/sidechain local error meanwhile got ~2x BETTER
        (0.543/0.669 vs 1.017/1.707 at dump 6), i.e. the term buys backbone
        accuracy by spending marker accuracy.

        Mechanism -- this is a shape problem, not a weight problem.  The four
        summed sigmoids make the restoring force NON-MONOTONIC in the error: it
        peaks near each threshold and nearly vanishes between them (normalized
        force ~4e-4 at 3.0 A vs 1.0 at the peaks).  Two consequences for
        markers, which must reach exactly 0:
          - near d=0 the 0.5 A sigmoid is already saturated on the good side, so
            the measured gradient at d=0.001 is 26x SMALLER than at its d=1 peak.
            The term cannot supply the last fraction of an angstrom.
          - the dead zone near 3.0 A acts as an attractor.  Observed anch_d ran
            2.45 -> 3.04 -> 2.91 -> 2.78 -> 2.46: markers parked in the trough
            and only crawled out under loss_cdr's monotonic pull.
        Re-tuning smooth_lddt_weight fixes neither -- scaling a force whose zeros
        sit in the wrong places leaves the zeros in place.  The measured gradient
        ratio was 0.654, already BELOW loss_cdr, so it was not over-weighted.

        If cross-loop geometry ever does need explicit supervision (e.g. once FR
        is unsupervised, or if loop-to-antigen packing enters scope), exclude
        marker slots from `valid_mask` rather than reviving it as-is: markers are
        268 of the 658 CDR atoms here yet the coincident pairs they add are only
        0.79% of the pair set (825/104444), so dropping them costs little signal.
        """
        bsz = pred_coords.shape[0]
        pred_flat = pred_coords.reshape(bsz, -1, 3)
        true_flat = true_coords.reshape(bsz, -1, 3)
        valid_mask = (cdr_mask.unsqueeze(-1) & atom14_mask).reshape(bsz, -1)
        lddt_list = []
        for i in range(bsz):
            mask_i = valid_mask[i]
            if mask_i.sum() < 2:
                lddt_list.append(pred_coords.new_tensor(1.0))
                continue
            pred_i = pred_flat[i][mask_i]  # [N_valid, 3]
            true_i = true_flat[i][mask_i]  # [N_valid, 3]
            true_dists = torch.cdist(true_i, true_i)
            pred_dists = torch.cdist(pred_i, pred_i)
            pair_mask = true_dists < cutoff
            pair_mask = torch.triu(pair_mask, diagonal=1)
            if pair_mask.sum() == 0:
                lddt_list.append(pred_coords.new_tensor(1.0))
                continue
            dist_diff = torch.abs(pred_dists[pair_mask] - true_dists[pair_mask])
            score = (
                torch.sigmoid(sharpness * (0.5 - dist_diff))
                + torch.sigmoid(sharpness * (1.0 - dist_diff))
                + torch.sigmoid(sharpness * (2.0 - dist_diff))
                + torch.sigmoid(sharpness * (4.0 - dist_diff))
            ) / 4.0
            lddt_list.append(score.mean())
        lddt = torch.stack(lddt_list).mean()
        return 1.0 - lddt

    def _prepare_cdr_atom_masks(
        self,
        loop_atom_valid_mask,
        loop_atom_supervise_mask,
        loop_valid_res_mask,
        loop_type_target,
        device,
    ):
        if loop_atom_valid_mask.ndim == 3:
            loop_atom_valid_mask = loop_atom_valid_mask.unsqueeze(0)
        if loop_atom_supervise_mask.ndim == 3:
            loop_atom_supervise_mask = loop_atom_supervise_mask.unsqueeze(0)
        if loop_valid_res_mask.ndim == 2:
            loop_valid_res_mask = loop_valid_res_mask.unsqueeze(0)
        if loop_type_target.ndim == 2:
            loop_type_target = loop_type_target.unsqueeze(0)

        physical_mask = loop_atom_valid_mask.to(device=device, dtype=torch.bool)
        supervise_mask = loop_atom_supervise_mask.to(device=device, dtype=torch.bool)
        valid_residue = loop_valid_res_mask.to(device=device, dtype=torch.bool)
        loop_type_target = loop_type_target.to(device=device, dtype=torch.long)

        expected_prefix = physical_mask.shape[:-1]
        if supervise_mask.shape != physical_mask.shape:
            raise ValueError("CDR physical and supervision masks have different shapes")
        if valid_residue.shape != expected_prefix or loop_type_target.shape != expected_prefix:
            raise ValueError("CDR residue mask or type target has an unexpected shape")

        invalid_type = valid_residue & (
            (loop_type_target < 0) | (loop_type_target >= len(RESD_NAMES_1C))
        )
        if bool(invalid_type.any()):
            raise ValueError("Valid CDR residue is missing a residue-type target")

        nominal_mask_table, _, _ = self._atom14_loss_tables(device)
        safe_type_target = loop_type_target.clamp(0, len(RESD_NAMES_1C) - 1)
        nominal_physical_mask = nominal_mask_table[safe_type_target]
        nominal_physical_mask = nominal_physical_mask & valid_residue.unsqueeze(-1)
        physical_mask = physical_mask & nominal_physical_mask
        supervise_mask = supervise_mask & valid_residue.unsqueeze(-1)

        atom_indices = torch.arange(14, device=device)
        backbone_mask = physical_mask & (atom_indices < 4)
        sidechain_mask = physical_mask & (atom_indices >= 4)
        virtual_mask = supervise_mask & ~nominal_physical_mask
        return physical_mask, valid_residue, backbone_mask, sidechain_mask, virtual_mask

    def _compute_full_bond_loss(
        self,
        pred_loop_local,
        clean_loop_local,
        physical_mask,
        loop_valid_res_mask,
        loop_type_target,
        cdr_scale=None,
    ):
        """Bond-length MSE, expressed in NORMALIZED length units.

        Why cdr_scale enters here (2026-09-01).  The network head emits a
        dimensionless `x0_norm`; fr_cdr_blocks.py:855 turns it physical with
        `pred_x0_local = x0_norm * cdr_scale + cdr_mu`.  _cdr_grouped_atom_mse
        therefore divides by cdr_scale**2, which makes it exactly the MSE in
        x0_norm space -- that is the EDM/Karras preconditioning, and it is what
        keeps the gradient reaching coord_head independent of the data scale.

        Bond length is a HOMOGENEOUS degree-2 function of the coordinates, so it
        has the same normalized form: with s = cdr_scale, d_norm = d / s and
        (d_norm - d*_norm)**2 = (d - d*)**2 / s**2.  Dividing by s**2 is thus an
        exact change of units, not a fudge factor.

        It matters because s**2 sits in the GRADIENT ratio, not just in the loss
        value.  With u = x0_norm and x = s*u,
            d(loss_cdr)/du  ~ err_phys / s
            d(loss_bond)/du ~ s * bond_err
        so |g_bond| / |g_cdr| = s**2 * bond_err / err_phys.  At s=6 that is a
        36x amplification: the 2026-09-01 bond_on run drove C=O error from
        0.218 A down to 0.036 A while the CDR local error degraded 0.275 ->
        1.072 A and fitted aar_cdr collapsed 0.83 -> 0.11.  Dividing by s**2
        cancels the factor exactly and leaves a ratio of physical quantities.

        NOTE this rule generalizes ONLY to terms homogeneous in the coordinates.
        loss_smooth_lddt has hard-coded angstrom thresholds (0.5/1/2/4 and
        cutoff=15) so rescaling its input would silently redefine the metric,
        and loss_seq is not a function of coordinates at all.  Those two must be
        weighted by measured gradient norm instead.

        `cdr_scale=None` keeps the old physical-angstrom behaviour, so the
        returned diagnostic components can still be read in A**2 if needed.
        """
        if clean_loop_local.ndim == 4:
            clean_loop_local = clean_loop_local.unsqueeze(0)
        device = pred_loop_local.device
        with torch.autocast(device_type=device.type, enabled=False):
            pred_loop_local = pred_loop_local.float()
            clean_loop_local = clean_loop_local.to(device=device, dtype=torch.float32)
            physical_mask = physical_mask.to(device=device, dtype=torch.bool)
            loop_valid_res_mask = loop_valid_res_mask.to(device=device, dtype=torch.bool)
            loop_type_target = loop_type_target.to(device=device, dtype=torch.long)

            pred_backbone_dist = torch.stack(
                [
                    torch.linalg.vector_norm(pred_loop_local[..., 0, :] - pred_loop_local[..., 1, :], dim=-1),
                    torch.linalg.vector_norm(pred_loop_local[..., 1, :] - pred_loop_local[..., 2, :], dim=-1),
                    torch.linalg.vector_norm(pred_loop_local[..., 2, :] - pred_loop_local[..., 3, :], dim=-1),
                ],
                dim=-1,
            )
            true_backbone_dist = torch.stack(
                [
                    torch.linalg.vector_norm(clean_loop_local[..., 0, :] - clean_loop_local[..., 1, :], dim=-1),
                    torch.linalg.vector_norm(clean_loop_local[..., 1, :] - clean_loop_local[..., 2, :], dim=-1),
                    torch.linalg.vector_norm(clean_loop_local[..., 2, :] - clean_loop_local[..., 3, :], dim=-1),
                ],
                dim=-1,
            )
            backbone_mask = torch.stack(
                [
                    physical_mask[..., 0] & physical_mask[..., 1],
                    physical_mask[..., 1] & physical_mask[..., 2],
                    physical_mask[..., 2] & physical_mask[..., 3],
                ],
                dim=-1,
            )

            pred_peptide_dist = torch.linalg.vector_norm(
                pred_loop_local[:, :, :-1, 2] - pred_loop_local[:, :, 1:, 0],
                dim=-1,
            )
            true_peptide_dist = torch.linalg.vector_norm(
                clean_loop_local[:, :, :-1, 2] - clean_loop_local[:, :, 1:, 0],
                dim=-1,
            )
            peptide_mask = (
                loop_valid_res_mask[:, :, :-1]
                & loop_valid_res_mask[:, :, 1:]
                & physical_mask[:, :, :-1, 2]
                & physical_mask[:, :, 1:, 0]
            )
            peptide_error = F.pad(
                (pred_peptide_dist - true_peptide_dist).square(), (0, 1)
            )
            peptide_mask = F.pad(peptide_mask, (0, 1), value=False)

            backbone_error = torch.cat(
                [
                    (pred_backbone_dist - true_backbone_dist).square(),
                    peptide_error.unsqueeze(-1),
                ],
                dim=-1,
            )
            backbone_mask = torch.cat(
                [backbone_mask, peptide_mask.unsqueeze(-1)], dim=-1
            )

            _, edge_indices_table, edge_mask_table = self._atom14_loss_tables(device)
            safe_type_target = loop_type_target.clamp(0, len(RESD_NAMES_1C) - 1)
            sidechain_edge_indices = edge_indices_table[safe_type_target]
            sidechain_edge_mask = (
                edge_mask_table[safe_type_target]
                & loop_valid_res_mask.unsqueeze(-1)
            )

            n_edges = sidechain_edge_indices.shape[-2]
            expanded_physical_mask = physical_mask.unsqueeze(3).expand(
                -1, -1, -1, n_edges, -1
            )
            edge_endpoint_mask = torch.gather(
                expanded_physical_mask, dim=4, index=sidechain_edge_indices
            )
            sidechain_edge_mask = sidechain_edge_mask & edge_endpoint_mask.all(dim=-1)

            pred_endpoints = self._gather_edge_endpoints(
                pred_loop_local, sidechain_edge_indices
            )
            true_endpoints = self._gather_edge_endpoints(
                clean_loop_local, sidechain_edge_indices
            )
            pred_sidechain_dist = torch.linalg.vector_norm(
                pred_endpoints[..., 0, :] - pred_endpoints[..., 1, :], dim=-1
            )
            true_sidechain_dist = torch.linalg.vector_norm(
                true_endpoints[..., 0, :] - true_endpoints[..., 1, :], dim=-1
            )
            sidechain_error = (pred_sidechain_dist - true_sidechain_dist).square()

            # Change of units to x0_norm space -- see the docstring.  Applied to
            # the squared errors so it lands on both components and on `total`.
            if cdr_scale is not None:
                s2 = (
                    cdr_scale.to(device=device, dtype=torch.float32)
                    .reshape(-1, 1, 1, 1)
                    .square()
                    .clamp_min(1e-8)
                )
                backbone_error = backbone_error / s2
                sidechain_error = sidechain_error / s2

            backbone_per_sample, backbone_active = self._masked_per_residue_mean(
                backbone_error, backbone_mask
            )
            sidechain_per_sample, sidechain_active = self._masked_per_residue_mean(
                sidechain_error, sidechain_edge_mask
            )
            total, components = self._aggregate_active_groups(
                [backbone_per_sample, sidechain_per_sample],
                [backbone_active, sidechain_active],
            )
            return total, components[0], components[1]

    def _cdr_grouped_atom_mse(
        self,
        pred_loop_local,
        clean_loop_local,
        backbone_mask,
        sidechain_mask,
        virtual_mask,
        cdr_scale,
    ):
        if clean_loop_local.ndim == 4:
            clean_loop_local = clean_loop_local.unsqueeze(0)
        device = pred_loop_local.device
        with torch.autocast(device_type=device.type, enabled=False):
            clean_loop_local = clean_loop_local.to(device=device, dtype=torch.float32)
            c_scale = cdr_scale.to(device=device, dtype=torch.float32).view(-1, 1, 1, 1)
            atom_error = (
                (pred_loop_local.float() - clean_loop_local).square().mean(dim=-1)
                / c_scale.square().clamp_min(1e-8)
            )

            group_masks = [backbone_mask, sidechain_mask, virtual_mask]
            group_values = []
            group_active = []
            for group_mask in group_masks:
                per_sample, sample_active = self._masked_per_residue_mean(
                    atom_error, group_mask
                )
                group_values.append(per_sample)
                group_active.append(sample_active)

            total, components = self._aggregate_active_groups(
                group_values, group_active
            )
            return total, components[0], components[1], components[2]

    # ------------------------------------------------------------------
    # Scheme B: sequence-head cross-entropy, AUXILIARY supervision only.
    # Residue type is decoded from atom14 virtual-atom geometry (marker
    # encoding), never from these logits; the head exists to give the trunk a
    # direct type signal.  Do not treat it as a decode path.
    #
    # Leak status (re-verified 2026-09-01, supersedes an earlier note here that
    # claimed an n_real leak): CDRLoopHead's `loop_atom_valid_mask` parameter is
    # bound to loop_atom_supervise_mask at structure_module.py:270, and that mask
    # is loop_valid_res_mask expanded to all 14 slots (diffuser.py:358) -- it is
    # all-True for every valid residue and carries NO per-residue atom count.
    # So n_real does not reach the network.  token_feat is gated only by
    # valid_res_mask, i.e. this head inherits the same true_len leak as the CDR
    # coordinate path and adds nothing new; that leak is accepted for the
    # fixed-length stage.
    # ------------------------------------------------------------------
    def _cdr_sequence_ce_loss(
        self,
        seq_logits: torch.Tensor,        # [B, n_loop, lmax, 20]
        loop_type_target: torch.Tensor,  # [B, n_loop, lmax] codebook idx, -100 ignore
    ) -> torch.Tensor:
        zero = seq_logits.new_tensor(0.0)
        if seq_logits is None or loop_type_target is None:
            return zero
        if seq_logits.ndim == 3:
            seq_logits = seq_logits.unsqueeze(0)
        if loop_type_target.ndim == 2:
            loop_type_target = loop_type_target.unsqueeze(0)
        n_cls = seq_logits.shape[-1]
        tgt = loop_type_target.to(device=seq_logits.device, dtype=torch.long)
        if not bool((tgt != -100).any()):
            return zero
        return F.cross_entropy(
            seq_logits.reshape(-1, n_cls), tgt.reshape(-1), ignore_index=-100
        )

    # ==========================================
    # Backbone loss: x0-space MSE (trsl) + geodesic (rota), sigma-independent
    # ==========================================
    def _backbone_mse_layer(self, inputs, pre_trsl, pre_rota):
        meta = inputs['anchor_frame_meta']
        device = pre_trsl.device
        with torch.autocast(device_type=device.type, enabled=False):
            tgt_trsl = meta['trsl_orig'].to(device=device, dtype=torch.float32).view(-1, 3)
            tgt_rota = meta['rota_orig'].to(device=device, dtype=torch.float32)
            sigma_data = meta['trsl_scale'].to(
                device=device, dtype=torch.float32
            ).view(-1, 1).clamp_min(1e-4)

            loss_trsl = (((pre_trsl.float().view(-1, 3) - tgt_trsl) / sigma_data) ** 2).sum(-1).mean()

            pre_r = pre_rota.to(device=device, dtype=torch.float32)
            relative = torch.matmul(pre_r.transpose(-1, -2), tgt_rota)
            skew = 0.5 * torch.stack(
                [
                    relative[..., 2, 1] - relative[..., 1, 2],
                    relative[..., 0, 2] - relative[..., 2, 0],
                    relative[..., 1, 0] - relative[..., 0, 1],
                ],
                dim=-1,
            )
            sin_angle = torch.linalg.norm(skew, dim=-1)
            cos_angle = (
                (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
            ).clamp(-1.0, 1.0)
            angle = torch.atan2(sin_angle, cos_angle)
            loss_rota = angle.square().mean()

            loss_backbone = 2.0 * loss_rota + loss_trsl
            return loss_backbone, loss_trsl, loss_rota


    # ----------------------------------------------------------------
    # main loss
    # ----------------------------------------------------------------
    def _aligned_backbone_cdr_vio_loss(
            self,
            inputs: Dict,
            outputs: Dict,
        ) -> Dict:
            # 1. Extract predictions and ground truth targets
            pred = outputs["3d"]["cord"][-1]
            batch_size, seq_len = pred.shape[:2]

            atom14_tgt = self._ensure_batched(
                inputs.get("cords_atom14", inputs["cord-o"]), ndim_no_batch=3
            ).to(device=pred.device, dtype=pred.dtype)

            cmsk = self._ensure_batched(inputs["cmsk-p"], ndim_no_batch=2).to(
                device=pred.device, dtype=torch.bool
            )

            cdr_mask = self._normalize_res_mask(
                inputs["cdr_mask"], batch_size, seq_len
            ).to(pred.device)

            # 2. Compute structure-level metric losses
            loss_smooth_lddt = self._cdr_smooth_lddt_loss(pred, atom14_tgt, cmsk, cdr_mask)

            # 3. Compute layer-wise CDR loop and closure losses
            loop_atom_supervise_mask = inputs["loop_atom_supervise_mask"]
            loop_atom_physical_mask = inputs["loop_atom_valid_mask"]
            loop_valid_res_mask = inputs["loop_valid_res_mask"]
            loop_cords_list = outputs["3d"]["loop_cords"]
            clean_loop_local_gt = inputs["clean_loop_local_coords"]

            atom14_type_target = inputs.get("atom14_type_target")
            if atom14_type_target is None:
                raise ValueError("atom14_type_target is required for grouped CDR losses")
            loop_type_target = self._gather_full_tensor_to_loops(
                atom14_type_target,
                inputs["loop_global_res_indices"],
                loop_valid_res_mask,
                fill_value=-100,
            ).to(device=pred.device, dtype=torch.long)

            (
                loop_atom_physical_mask,
                loop_valid_res_mask,
                backbone_atom_mask,
                sidechain_atom_mask,
                virtual_atom_mask,
            ) = self._prepare_cdr_atom_masks(
                loop_atom_physical_mask,
                loop_atom_supervise_mask,
                loop_valid_res_mask,
                loop_type_target,
                pred.device,
            )

            n_layers = len(loop_cords_list)
            if n_layers == 0:
                raise ValueError("No structure-module layer outputs")

            loss_cdr = pred.new_tensor(0.0)
            loss_cdr_backbone = pred.new_tensor(0.0)
            loss_cdr_sidechain = pred.new_tensor(0.0)
            loss_cdr_virtual = pred.new_tensor(0.0)
            layer_weight_sum = 0.0

            # Weight layers increasingly (1.0 for layer 0, 2.0 for layer 1, etc.)
            for layer_idx, pred_loop_local in enumerate(loop_cords_list):
                layer_weight = float(layer_idx + 1)
                layer_weight_sum += layer_weight

                (
                    loss_cdr_layer,
                    loss_cdr_backbone_layer,
                    loss_cdr_sidechain_layer,
                    loss_cdr_virtual_layer,
                ) = self._cdr_grouped_atom_mse(
                    pred_loop_local,
                    clean_loop_local_gt,
                    backbone_atom_mask,
                    sidechain_atom_mask,
                    virtual_atom_mask,
                    inputs["cdr_meta"]["cdr_scale"],
                )
                loss_cdr = loss_cdr + layer_weight * loss_cdr_layer
                loss_cdr_backbone = (
                    loss_cdr_backbone + layer_weight * loss_cdr_backbone_layer
                )
                loss_cdr_sidechain = (
                    loss_cdr_sidechain + layer_weight * loss_cdr_sidechain_layer
                )
                loss_cdr_virtual = (
                    loss_cdr_virtual + layer_weight * loss_cdr_virtual_layer
                )

            loss_cdr = loss_cdr / layer_weight_sum
            loss_cdr_backbone = loss_cdr_backbone / layer_weight_sum
            loss_cdr_sidechain = loss_cdr_sidechain / layer_weight_sum
            loss_cdr_virtual = loss_cdr_virtual / layer_weight_sum

            loss_bond, loss_bond_backbone, loss_bond_sidechain = (
                self._compute_full_bond_loss(
                    loop_cords_list[-1],
                    clean_loop_local_gt,
                    loop_atom_physical_mask,
                    loop_valid_res_mask,
                    loop_type_target,
                    cdr_scale=inputs["cdr_meta"]["cdr_scale"],
                )
            )

            loss_trsl_residual, loss_rota_residual, rotation_diag = self._fr_residual_loss(
                inputs, outputs
            )
            loss_global_backbone = pred.new_tensor(0.0)
            loss_trsl = pred.new_tensor(0.0)
            loss_rota = pred.new_tensor(0.0)
            fr_layer_weight_sum = 0.0
            for layer_idx, (pred_trsl, pred_rota) in enumerate(
                zip(outputs["3d"]["trsl"], outputs["3d"]["rota"])
            ):
                layer_weight = float(layer_idx + 1)
                fr_layer_weight_sum += layer_weight
                layer_backbone, layer_trsl, layer_rota = self._backbone_mse_layer(
                    inputs, pred_trsl, pred_rota
                )
                loss_global_backbone = (
                    loss_global_backbone + layer_weight * layer_backbone
                )
                loss_trsl = loss_trsl + layer_weight * layer_trsl
                loss_rota = loss_rota + layer_weight * layer_rota
            loss_global_backbone = loss_global_backbone / fr_layer_weight_sum
            loss_trsl = loss_trsl / fr_layer_weight_sum
            loss_rota = loss_rota / fr_layer_weight_sum

            loss_backbone = (
                loss_trsl_residual
                + loss_rota_residual
                + self.cfg.fr_global_aux_weight * loss_global_backbone
            )
            loss_vio = torch.zeros_like(loss_backbone)

            # 6. Calculate SNR (Signal-to-Noise Ratio) based weights
            if self.cfg.use_snr_weight:
                sigma_t = inputs["cdr_meta"]["cdr_sigma"].to(
                    device=pred.device, dtype=pred.dtype
                ).reshape(-1).clamp_min(1e-6)
                sigma_data = inputs["cdr_meta"]["cdr_scale"].to(
                    device=pred.device, dtype=pred.dtype
                ).reshape(-1).clamp_min(1e-6)
                snr = (sigma_data / sigma_t) ** 2
                w_cdr = (
                    torch.clamp(snr, max=self.cfg.snr_gamma) / (snr + 1.0)
                ).mean()
            else:
                w_cdr = pred.new_tensor(1.0)

            # Scheme B: sequence-head CE (inert unless seq_head_weight > 0).
            loss_seq = pred.new_tensor(0.0)
            seq_logits = outputs["3d"].get("seq_logits")
            if self.cfg.seq_head_weight > 0.0: # and seq_logits is not None and "atom14_type_target" in inputs
                loss_seq = self._cdr_sequence_ce_loss(seq_logits, loop_type_target)

            # 7. Loss aggregation with config weights

            total = (
                self.cfg.backbone_weight * loss_backbone
                # AB2: loss_cdr enabled (weight lowered to 1.0 in the config).
                # It is pure LOCAL-frame MSE, so it does NOT flow into the FR
                # pose: its only interaction with FR is gradient competition in
                # the shared trunk.  bond / smooth_lddt / seq stay off until FR
                # is confirmed non-degraded -- see the note on _cdr_smooth_lddt_loss.
                + self.cfg.cdr_all_atom_weight * w_cdr * loss_cdr
                + self.cfg.bond_weight * loss_bond
                # smooth_lddt stays OFF.  It was enabled at weight 1.0 on
                # 2026-09-01 (gradient ratio 0.654 vs loss_cdr, so the weight was
                # NOT the problem) and the run was stopped early: it wrecks the
                # type channel.  See _cdr_smooth_lddt_loss for the mechanism and
                # the numbers.  Do not re-enable by re-tuning the weight.
                # + self.cfg.smooth_lddt_weight * loss_smooth_lddt
                + self.cfg.seq_head_weight * loss_seq
                # + self.cfg.vio_weight * loss_vio
            )
            self._log_active_loss_terms(locals())

            if self.idx_save % 100 == 0:
                import time
                ts = int(time.time())
                rota_meta = inputs["anchor_frame_meta"]
                torch.save({
                    'perturb': inputs['cord-p'],
                    'pre': pred,
                    'clean': atom14_tgt,
                    'step': inputs['step'],
                    # 'sigma_raw': inputs['sigama_t']['sigma_raw'],
                    # 'fr_rota_rms': rota_meta['fr_rota_rms'],
                    # 'fr_igso3_eps': rota_meta['fr_igso3_eps'],
                    # 'fr_rota_is_haar': rota_meta['fr_rota_is_haar'],
                    # 'rota_xt': rota_meta['rota_xt'],
                }, f'/root/private_data/luog/codex/IgGM2/see/seefile/S0907_100_{ts}.pt')
            self.idx_save += 1

            return {
                "loss": total,
                "loss_viol": loss_vio,
                "loss_backbone": loss_backbone,
                "loss_cdr": loss_cdr,
                "loss_cdr_backbone": loss_cdr_backbone,
                "loss_cdr_sidechain": loss_cdr_sidechain,
                "loss_cdr_virtual": loss_cdr_virtual,
                "loss_smooth_lddt": loss_smooth_lddt,
                "loss_bond": loss_bond,
                "loss_bond_backbone": loss_bond_backbone,
                "loss_bond_sidechain": loss_bond_sidechain,
                "loss_trsl": loss_trsl,
                "loss_rota": loss_rota,

                "loss_trsl_residual": loss_trsl_residual,
                "loss_rota_residual": loss_rota_residual,
                "loss_seq": loss_seq,

                "w_cdr": w_cdr,
                "rotation_diag": rotation_diag,
            }
