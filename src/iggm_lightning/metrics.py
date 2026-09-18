# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""Evaluation metrics used by the IgGM Lightning validation/test loop."""

from __future__ import annotations

import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
from tmtools import tm_align

from IgGM.protein.prot_constants import ATOM_NAMES_PER_RESD, RESD_MAP_1TO3


@dataclass
class MetricConfig:
    dockq_threshold: float = 0.23


class StructureMetrics:
    """Metrics driven by external libraries (DockQ + tmtools)."""

    LOOP_NAMES = ("H1", "H2", "H3", "L1", "L2", "L3")

    def __init__(self, cfg: MetricConfig | None = None) -> None:
        self.cfg = cfg or MetricConfig()
        self._warned_messages: set[str] = set()

    def _warn_once(self, message: str) -> None:
        if message in self._warned_messages:
            return
        self._warned_messages.add(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    @staticmethod
    def _rmsd(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        tgt = tgt.float()
        return torch.sqrt(((pred - tgt) ** 2).sum(dim=-1).mean().clamp_min(1e-8))

    # @staticmethod
    # def _kabsch_align(pred: torch.Tensor, tgt: torch.Tensor):
    #     pred = pred.float()
    #     tgt = tgt.float()

    #     pred_mean = pred.mean(dim=0, keepdim=True)
    #     tgt_mean = tgt.mean(dim=0, keepdim=True)

    #     pred_c = pred - pred_mean
    #     tgt_c = tgt - tgt_mean

    #     h = (pred_c.transpose(0, 1) @ tgt_c).float()
    #     u, _, vh = torch.linalg.svd(h, full_matrices=False)
    #     v = vh.transpose(-2, -1)

    #     r = (v @ u.transpose(0, 1)).float()
    #     if torch.det(r.float()) < 0:
    #         v = v.clone()
    #         v[:, -1] *= -1
    #         r = (v @ u.transpose(0, 1)).float()

    #     pred_aligned = pred_c @ r + tgt_mean
    #     return pred_aligned.float(), tgt.float()

    @staticmethod
    def _kabsch_align(pred: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Align pred to tgt using the Kabsch algorithm (row-vector convention)."""
        orig_dtype = pred.dtype
        device_type = "cuda" if pred.is_cuda else "cpu"
        
        # 强制关闭自动混合精度 (AMP)，防止矩阵乘法 (@) 将数据打回 bfloat16
        with torch.autocast(device_type=device_type, enabled=False):
            pred = pred.to(torch.float32)
            tgt = tgt.to(torch.float32)

            # 使用 dim=-2 兼容 (B, N, 3) 和 (N, 3)
            pred_mean = pred.mean(dim=-2, keepdim=True)
            tgt_mean = tgt.mean(dim=-2, keepdim=True)

            pred_c = pred - pred_mean
            tgt_c = tgt - tgt_mean

            # 计算协方差矩阵 H (pred^T @ tgt)
            h = pred_c.transpose(-2, -1) @ tgt_c
            u, _, vh = torch.linalg.svd(h, full_matrices=False)

            # 反射校正 (Reflection correction)，同时兼容 batch 维度
            d = torch.ones_like(h[..., 0])  # Shape: (B, 3) or (3,)
            d[..., -1] = torch.sign(torch.det(u @ vh))
            
            # 使用 diag_embed 构造对角矩阵，支持 Batched 运算
            r = u @ torch.diag_embed(d) @ vh

            # 应用旋转和平移
            pred_aligned = pred_c @ r + tgt_mean

        return pred_aligned.to(orig_dtype), tgt.to(orig_dtype)
    

    @staticmethod
    def _extract_loop_indices(cdr_sequences: Mapping[str, List[int]] | None, seq_lengths: Mapping[str, int] | None) -> Dict[str, List[int]]:
        cdr_sequences = cdr_sequences or {}
        seq_lengths = seq_lengths or {}
        h_len = int(seq_lengths.get("H", 0))

        def _get(name: str, offset: int = 0) -> List[int]:
            out: List[int] = []
            for idx_1b in cdr_sequences.get(name, []):
                idx = int(idx_1b) - 1 + offset
                if idx >= 0:
                    out.append(idx)
            return sorted(set(out))

        return {
            "H1": _get("cdr_H1", offset=0),
            "H2": _get("cdr_H2", offset=0),
            "H3": _get("cdr_H3", offset=0),
            "L1": _get("cdr_L1", offset=h_len),
            "L2": _get("cdr_L2", offset=h_len),
            "L3": _get("cdr_L3", offset=h_len),
        }

    @staticmethod
    def _aar(pred_seq: str, true_seq: str) -> float:
        if not pred_seq or not true_seq:
            return 0.0
        n = min(len(pred_seq), len(true_seq))
        if n == 0:
            return 0.0
        return sum(1 for a, b in zip(pred_seq[:n], true_seq[:n]) if a == b) / n

    @staticmethod
    def _safe_loop_metric(vals: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        if not vals:
            return torch.tensor(float("nan"), dtype=torch.float32, device=device)
        return torch.stack(vals).mean()

    @staticmethod
    def _is_finite_dict(metrics: Dict[str, float], keys: List[str]) -> bool:
        for k in keys:
            v = metrics.get(k, float("nan"))
            if not np.isfinite(float(v)):
                return False
        return True

    @staticmethod
    def _write_minimal_ca_pdb(path: Path, ca: torch.Tensor, asym_id: torch.Tensor) -> None:
        chain_symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        uniq = torch.unique(asym_id).detach().cpu().tolist()
        chain_map = {int(cid): chain_symbols[i % len(chain_symbols)] for i, cid in enumerate(uniq)}
        lines: List[str] = []
        serial = 1
        resi_count = {int(cid): 1 for cid in uniq}
        for i in range(ca.shape[0]):
            cid = int(asym_id[i].item())
            chain_id = chain_map[cid]
            x, y, z = [float(v) for v in ca[i].detach().cpu().tolist()]
            resi = resi_count[cid]
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {chain_id}{resi:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
            )
            serial += 1
            resi_count[cid] += 1
        lines.append("TER")
        lines.append("END")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _calc_dockq_legacy_ca(self, pred_ca: torch.Tensor, tgt_ca: torch.Tensor, asym_id: torch.Tensor) -> Dict[str, float]:
        """Legacy CA-only path retained for diagnostics; not used for reporting."""
        with tempfile.TemporaryDirectory(prefix="iggm_dockq_") as td:
            pred_pdb = Path(td) / "pred.pdb"
            tgt_pdb = Path(td) / "native.pdb"
            self._write_minimal_ca_pdb(pred_pdb, pred_ca, asym_id)
            self._write_minimal_ca_pdb(tgt_pdb, tgt_ca, asym_id)

            from DockQ.DockQ import load_PDB, run_on_all_native_interfaces  # type: ignore

            model = load_PDB(str(pred_pdb))
            native = load_PDB(str(tgt_pdb))
            result = run_on_all_native_interfaces(model, native)

            if isinstance(result, dict) and "best_result" in result and isinstance(result["best_result"], dict):
                result = result["best_result"]
            if not isinstance(result, dict):
                raise RuntimeError("DockQ python API returned non-dict result")
            required = ("DockQ", "iRMS", "LRMS", "fnat")
            if not all(k in result for k in required):
                raise RuntimeError(f"DockQ python API missing keys: {required}")

            return {
                "dockq": float(result["DockQ"]),
                "fnat": float(result["fnat"]),
                "lrms": float(result["LRMS"]),
                "irms": float(result["iRMS"]),
            }

    @staticmethod
    def _nominal_atom14_mask(seq: str, device: torch.device) -> torch.Tensor:
        mask = torch.zeros((len(seq), 14), dtype=torch.bool, device=device)
        for idx, aa in enumerate(seq):
            if aa not in RESD_MAP_1TO3:
                raise ValueError(f"Unsupported residue {aa!r} at position {idx}")
            mask[idx, :len(ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[aa]])] = True
        return mask

    @staticmethod
    def _as_bool_vector(mask: torch.Tensor, length: int, name: str) -> torch.Tensor:
        mask = mask.detach().to(dtype=torch.bool).reshape(-1)
        if mask.numel() != length:
            raise ValueError(f"{name} has {mask.numel()} values, expected {length}")
        return mask

    @classmethod
    def _write_full_atom_dockq_pdb(
        cls,
        path: Path,
        cord: torch.Tensor,
        atom_mask: torch.Tensor,
        seq: str,
        antibody_mask: torch.Tensor,
        antigen_mask: torch.Tensor,
    ) -> int:
        cord = cord.detach().float().cpu()
        atom_mask = atom_mask.detach().to(dtype=torch.bool).cpu()
        length = cord.shape[0]
        if cord.shape != (length, 14, 3) or atom_mask.shape != (length, 14):
            raise ValueError(
                f"DockQ atom14 shapes must be [L,14,3]/[L,14], got "
                f"{tuple(cord.shape)}/{tuple(atom_mask.shape)}"
            )
        if len(seq) != length:
            raise ValueError(f"DockQ sequence length {len(seq)} != coordinate length {length}")

        antibody_mask = cls._as_bool_vector(antibody_mask, length, "antibody_mask").cpu()
        antigen_mask = cls._as_bool_vector(antigen_mask, length, "antigen_mask").cpu()
        if torch.any(antibody_mask & antigen_mask):
            raise ValueError("antibody_mask and antigen_mask overlap")
        if not torch.all(antibody_mask | antigen_mask):
            raise ValueError("antibody_mask and antigen_mask do not cover every residue")
        ab_indices = torch.where(antibody_mask)[0]
        ag_indices = torch.where(antigen_mask)[0]
        if ab_indices.numel() == 0 or ag_indices.numel() == 0:
            raise ValueError("DockQ requires non-empty antibody and antigen partners")
        if not torch.equal(ab_indices, torch.arange(ab_indices.numel())):
            raise ValueError("antibody residues must form a contiguous prefix (H-L or H)")
        if not torch.equal(
            ag_indices,
            torch.arange(ab_indices.numel(), length),
        ):
            raise ValueError("antigen residues must form a contiguous suffix")

        active_coords = cord[atom_mask]
        if active_coords.numel() == 0 or not torch.isfinite(active_coords).all():
            raise ValueError("DockQ PDB contains no atoms or non-finite active coordinates")

        lines: List[str] = []
        serial = 1
        for chain_id, residue_indices in (("A", ag_indices), ("B", ab_indices)):
            for residue_number, residue_idx_tensor in enumerate(residue_indices, start=1):
                residue_idx = int(residue_idx_tensor)
                aa = seq[residue_idx]
                residue_name = RESD_MAP_1TO3[aa]
                atom_names = ATOM_NAMES_PER_RESD[residue_name]
                for atom_idx, atom_name in enumerate(atom_names):
                    if not bool(atom_mask[residue_idx, atom_idx]):
                        continue
                    x, y, z = cord[residue_idx, atom_idx].tolist()
                    element = atom_name[0]
                    lines.append(
                        f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} "
                        f"{chain_id}{residue_number:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          "
                        f"{element:>2s}"
                    )
                    serial += 1
            lines.append("TER")
        lines.append("END")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return serial - 1

    def _calc_dockq(
        self,
        pred_cord: torch.Tensor,
        tgt_cord: torch.Tensor,
        pred_seq: str,
        true_seq: str,
        pred_atom_mask: torch.Tensor,
        tgt_atom_mask: torch.Tensor,
        antibody_mask: torch.Tensor,
        antigen_mask: torch.Tensor,
    ) -> Dict[str, float]:
        with tempfile.TemporaryDirectory(prefix="iggm_dockq_") as temp_dir:
            pred_pdb = Path(temp_dir) / "pred.pdb"
            tgt_pdb = Path(temp_dir) / "native.pdb"
            pred_count = self._write_full_atom_dockq_pdb(
                pred_pdb, pred_cord, pred_atom_mask, pred_seq,
                antibody_mask, antigen_mask,
            )
            tgt_count = self._write_full_atom_dockq_pdb(
                tgt_pdb, tgt_cord, tgt_atom_mask, true_seq,
                antibody_mask, antigen_mask,
            )
            if pred_count != int(pred_atom_mask.sum()) or tgt_count != int(tgt_atom_mask.sum()):
                raise RuntimeError("DockQ PDB atom count does not match the physical atom mask")

            from DockQ.DockQ import load_PDB, run_on_chains  # type: ignore

            model = load_PDB(str(pred_pdb))
            native = load_PDB(str(tgt_pdb))
            result = run_on_chains(
                (model["A"], model["B"]),
                (native["A"], native["B"]),
                small_molecule=False,
            )
            required = ("DockQ", "iRMSD", "LRMSD", "fnat")
            if not isinstance(result, dict) or not all(k in result for k in required):
                raise RuntimeError(f"DockQ 2.1.3 result missing keys {required}")
            return {
                "dockq": float(result["DockQ"]),
                "fnat": float(result["fnat"]),
                "lrms": float(result["LRMSD"]),
                "irms": float(result["iRMSD"]),
            }

    @staticmethod
    def _nan_dockq() -> Dict[str, float]:
        return {key: float("nan") for key in ("dockq", "fnat", "lrms", "irms")}

    def _fallback_dockq(self, pred_ca: torch.Tensor, tgt_ca: torch.Tensor) -> Dict[str, float]:
        pred_aln, tgt_aln = self._kabsch_align(pred_ca, tgt_ca)
        ca_rmsd = float(self._rmsd(pred_aln, tgt_aln).item())
        # conservative fallback (no interface split available)
        dockq = 1.0 / (1.0 + (ca_rmsd / 8.5) ** 2)
        return {
            "dockq": float(dockq),
            "fnat": 0.0,
            "lrms": float(ca_rmsd),
            "irms": float(ca_rmsd),
        }

    @staticmethod
    def _calc_lddt(aligned_pred: np.ndarray, aligned_true: np.ndarray) -> float:
        # C-alpha lDDT style per-residue thresholds
        dist = np.linalg.norm(aligned_pred - aligned_true, axis=-1)
        score = (
            (dist < 0.5).astype(np.float32)
            + (dist < 1.0).astype(np.float32)
            + (dist < 2.0).astype(np.float32)
            + (dist < 4.0).astype(np.float32)
        ) / 4.0
        return float(np.mean(score))

    def _calc_tm_gdt_lddt(
        self,
        pred_ca: torch.Tensor,
        tgt_ca: torch.Tensor,
        pred_seq: str,
        true_seq: str,
    ) -> Dict[str, float]:
        pred_np = pred_ca.detach().cpu().numpy().astype(np.float64, copy=False)
        tgt_np = tgt_ca.detach().cpu().numpy().astype(np.float64, copy=False)
        result = tm_align(pred_np, tgt_np, pred_seq, true_seq)

        tm_val = getattr(result, "tm_norm_chain1", None)
        if tm_val is None:
            tm_val = getattr(result, "tm_norm_1", None)
        if tm_val is None:
            raise RuntimeError("tmtools output missing TM-score fields")

        aligned_pred = getattr(result, "coords1_aligned", None)
        aligned_true = getattr(result, "coords2", None)
        if aligned_pred is None or aligned_true is None:
            pred_aln_t, true_aln_t = self._kabsch_align(pred_ca, tgt_ca)
            aligned_pred = pred_aln_t.detach().cpu().numpy()
            aligned_true = true_aln_t.detach().cpu().numpy()

        dist = np.linalg.norm(aligned_pred - aligned_true, axis=-1)
        gdt_ts = float(np.mean([
            np.mean(dist <= 1.0),
            np.mean(dist <= 2.0),
            np.mean(dist <= 4.0),
            np.mean(dist <= 8.0),
        ]))
        lddt = self._calc_lddt(aligned_pred, aligned_true)

        return {
            "tm_score": float(tm_val),
            "gdt_ts": gdt_ts,
            "lddt": lddt,
        }

    def _fallback_tm_gdt_lddt(self, pred_ca: torch.Tensor, tgt_ca: torch.Tensor) -> Dict[str, float]:
        pred_aligned, tgt_aligned = self._kabsch_align(pred_ca, tgt_ca)
        dist = torch.norm(pred_aligned - tgt_aligned, dim=-1)
        n = max(int(pred_aligned.shape[0]), 1)
        d0 = max(0.5, 1.24 * ((max(n, 16) - 15) ** (1 / 3)) - 1.8)
        tm_score = float((1.0 / (1.0 + (dist / d0) ** 2)).mean().item())
        gdt_ts = float(torch.stack([(dist <= t).float().mean() for t in (1.0, 2.0, 4.0, 8.0)]).mean().item())
        lddt = self._calc_lddt(pred_aligned.detach().cpu().numpy(), tgt_aligned.detach().cpu().numpy())
        return {"tm_score": tm_score, "gdt_ts": gdt_ts, "lddt": float(lddt)}

    def compute_loop_metrics(
        self,
        pred_cord: torch.Tensor,
        tgt_cord: torch.Tensor,
        pred_seq: str,
        true_seq: str,
        cdr_h3_idx: List[int] | None = None,
        cdr_sequences: Optional[Mapping[str, List[int]]] = None,
        seq_lengths: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        pred_cord = pred_cord.float().detach()
        tgt_cord = tgt_cord.float().detach()
        device = pred_cord.device
        loop_map = self._extract_loop_indices(cdr_sequences, seq_lengths)
        if not any(loop_map.values()) and cdr_h3_idx:
            loop_map["H3"] = [i - 1 for i in cdr_h3_idx if i > 0]

        for_loop_rmsd: List[torch.Tensor] = []
        for_loop_aar: List[torch.Tensor] = []
        loop_metrics: Dict[str, torch.Tensor] = {}
        pred_seq_local = str(pred_seq)
        true_seq_local = str(true_seq)

        for loop_name in self.LOOP_NAMES:
            idxs = [
                idx for idx in loop_map.get(loop_name, [])
                if 0 <= idx < pred_cord.shape[0] and idx < tgt_cord.shape[0]
            ]
            if not idxs:
                loop_metrics[f"rmsd_{loop_name}"] = torch.tensor(
                    0.0 if loop_name.startswith("L") else float("nan"),
                    dtype=torch.float32,
                    device=device,
                )
                loop_metrics[f"aar_{loop_name}"] = torch.tensor(
                    0.0, dtype=torch.float32, device=device
                )
                continue

            idx = torch.tensor(idxs, device=device, dtype=torch.long)
            pred_loop = pred_cord[idx, :3].reshape(-1, 3)
            tgt_loop = tgt_cord[idx, :3].reshape(-1, 3)
            pred_loop_aln, tgt_loop_aln = self._kabsch_align(pred_loop, tgt_loop)
            rmsd_val = self._rmsd(pred_loop_aln, tgt_loop_aln)
            loop_metrics[f"rmsd_{loop_name}"] = rmsd_val
            for_loop_rmsd.append(rmsd_val)

            pred_loop_seq = "".join(
                pred_seq_local[idx] for idx in idxs if idx < len(pred_seq_local)
            )
            true_loop_seq = "".join(
                true_seq_local[idx] for idx in idxs if idx < len(true_seq_local)
            )
            aar_val = torch.tensor(
                self._aar(pred_loop_seq, true_loop_seq),
                dtype=torch.float32,
                device=device,
            )
            loop_metrics[f"aar_{loop_name}"] = aar_val
            for_loop_aar.append(aar_val)

        rmsd_h3 = loop_metrics.get("rmsd_H3")
        if rmsd_h3 is None or torch.isnan(rmsd_h3):
            pred_ca = pred_cord[:, 1]
            tgt_ca = tgt_cord[:, 1]
            pred_aligned, tgt_aligned = self._kabsch_align(pred_ca, tgt_ca)
            rmsd_h3 = self._rmsd(pred_aligned, tgt_aligned)

        return {
            "aar_loop_mean": self._safe_loop_metric(for_loop_aar, device=device),
            "rmsd_h3": rmsd_h3,
            "rmsd_loop_mean": self._safe_loop_metric(for_loop_rmsd, device=device),
            **loop_metrics,
        }

    def __call__(
        self,
        pred_cord: torch.Tensor,
        tgt_cord: torch.Tensor,
        pred_seq: str,
        true_seq: str,
        cdr_h3_idx: List[int] | None = None,
        asym_id: Optional[torch.Tensor] = None,
        cdr_sequences: Optional[Mapping[str, List[int]]] = None,
        seq_lengths: Optional[Mapping[str, int]] = None,
        native_atom_mask: Optional[torch.Tensor] = None,
        antibody_mask: Optional[torch.Tensor] = None,
        antigen_mask: Optional[torch.Tensor] = None,
        cdr_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pred_cord = pred_cord.float().detach()
        tgt_cord = tgt_cord.float().detach()

        pred_ca = pred_cord[:, 1]
        tgt_ca = tgt_cord[:, 1]

        if asym_id is None:
            asym_id = torch.zeros(pred_ca.shape[0], device=pred_ca.device, dtype=torch.long)
        if asym_id.ndim == 2:
            asym_id = asym_id[0]
        asym_id = asym_id.to(device=pred_ca.device)

        dockq_inputs = (native_atom_mask, antibody_mask, antigen_mask, cdr_mask)
        if all(value is not None for value in dockq_inputs):
            try:
                nominal_pred = self._nominal_atom14_mask(pred_seq, pred_cord.device)
                nominal_true = self._nominal_atom14_mask(true_seq, tgt_cord.device)
                original_mask = native_atom_mask.to(
                    device=pred_cord.device, dtype=torch.bool
                ).reshape(pred_cord.shape[0], 14)
                cdr_mask_vec = self._as_bool_vector(
                    cdr_mask, pred_cord.shape[0], "cdr_mask"
                ).to(device=pred_cord.device)
                pred_atom_mask = original_mask & nominal_pred
                pred_atom_mask[cdr_mask_vec] = nominal_pred[cdr_mask_vec]
                tgt_atom_mask = original_mask & nominal_true
                dockq_dict = self._calc_dockq(
                    pred_cord,
                    tgt_cord,
                    pred_seq,
                    true_seq,
                    pred_atom_mask,
                    tgt_atom_mask,
                    antibody_mask,
                    antigen_mask,
                )
                if not self._is_finite_dict(
                    dockq_dict, ["dockq", "fnat", "lrms", "irms"]
                ):
                    raise RuntimeError("DockQ returned non-finite metrics")
            except Exception as exc:
                self._warn_once(f"Official DockQ evaluation failed; reporting NaN: {exc}")
                dockq_dict = self._nan_dockq()
        else:
            self._warn_once(
                "DockQ requires native_atom_mask, antibody_mask, antigen_mask, and "
                "cdr_mask; reporting NaN because one or more are missing"
            )
            dockq_dict = self._nan_dockq()

        try:
            tm_dict = self._calc_tm_gdt_lddt(
                pred_ca, tgt_ca, pred_seq, true_seq
            )
        except Exception as exc:
            self._warn_once(f"tmtools.tm_align failed; using local fallback: {exc}")
            tm_dict = self._fallback_tm_gdt_lddt(pred_ca, tgt_ca)
        if not self._is_finite_dict(tm_dict, ["tm_score", "gdt_ts", "lddt"]):
            self._warn_once("tmtools returned non-finite values; using local fallback")
            tm_dict = self._fallback_tm_gdt_lddt(pred_ca, tgt_ca)

        loop_metrics = self.compute_loop_metrics(
            pred_cord,
            tgt_cord,
            pred_seq,
            true_seq,
            cdr_h3_idx,
            cdr_sequences,
            seq_lengths,
        )

        dockq = torch.tensor(dockq_dict["dockq"], dtype=torch.float32, device=pred_ca.device)
        fnat = torch.tensor(dockq_dict["fnat"], dtype=torch.float32, device=pred_ca.device)
        lrms = torch.tensor(dockq_dict["lrms"], dtype=torch.float32, device=pred_ca.device)
        irms = torch.tensor(dockq_dict["irms"], dtype=torch.float32, device=pred_ca.device)

        return {
            "aar": torch.tensor(self._aar(pred_seq, true_seq), dtype=torch.float32, device=pred_ca.device),
            "tm_score": torch.tensor(tm_dict["tm_score"], dtype=torch.float32, device=pred_ca.device),
            "gdt_ts": torch.tensor(tm_dict["gdt_ts"], dtype=torch.float32, device=pred_ca.device),
            "lddt": torch.tensor(tm_dict["lddt"], dtype=torch.float32, device=pred_ca.device),
            "dockq": dockq,
            "sr": (dockq >= self.cfg.dockq_threshold).float(),
            "fnat": fnat,
            "lrms": lrms,
            "irms": irms,
            **loop_metrics,
        }
