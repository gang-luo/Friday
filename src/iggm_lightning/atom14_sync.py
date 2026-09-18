"""Atom14 virtual-atom encoding/decoding utilities for CDR sequence synchronization.

Implementation note for this IgGM codebase:
- tensors are consumed in the repository's native n14-tf layout (variable real-atom prefix + padded tail),
  so virtual markers are written directly into padded tail slots;
- BoltzGen residue codebook is applied as (#N, #O) marker counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from IgGM.protein.prot_constants import (
    ATOM_NAMES_PER_RESD,
    RESD_MAP_1TO3,
    RESD_NAMES_1C,
)


@dataclass(frozen=True)
class BoltzResidueCode:
    """BoltzGen residue marker code: number of virtual atoms on backbone N and O."""

    n_on_n: int
    n_on_o: int

    @property
    def n_markers(self) -> int:
        return self.n_on_n + self.n_on_o


class Atom14SeqSync:
    """Build atom14 supervision and decode CDR sequence from BoltzGen marker counts."""

    _BOLTZ_CODEBOOK: Dict[str, BoltzResidueCode] = {
        "G": BoltzResidueCode(0, 10), "A": BoltzResidueCode(0, 9), "C": BoltzResidueCode(0, 8),
        "S": BoltzResidueCode(8, 0),  "P": BoltzResidueCode(0, 7), "T": BoltzResidueCode(3, 4),
        "V": BoltzResidueCode(7, 0),  "I": BoltzResidueCode(0, 6), "N": BoltzResidueCode(1, 5),
        "D": BoltzResidueCode(2, 4),  "L": BoltzResidueCode(4, 2), "M": BoltzResidueCode(6, 0),
        "Q": BoltzResidueCode(0, 5),  "E": BoltzResidueCode(2, 3), "K": BoltzResidueCode(5, 0),
        "H": BoltzResidueCode(0, 4),  "F": BoltzResidueCode(0, 3), "R": BoltzResidueCode(3, 0),
        "Y": BoltzResidueCode(0, 2),  "W": BoltzResidueCode(0, 0),
    }

    def __init__(self, decode_threshold: float = 1.0) -> None:
        self.decode_threshold = float(decode_threshold)
        self._n_real_dict: Dict[str, int] = {
            aa: len(ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[aa]])
            for aa in RESD_NAMES_1C
        }


    @staticmethod
    def _build_residue_meta() -> Dict[str, Dict[str, int]]:
        """Build per-residue metadata in native n14-tf layout."""
        out: Dict[str, Dict[str, int]] = {}
        for aa in RESD_NAMES_1C:
            res3 = RESD_MAP_1TO3[aa]
            atom_names = list(ATOM_NAMES_PER_RESD[res3])
            n_real = len(atom_names)
            if n_real > 14:
                raise ValueError(f"Unexpected >14 atoms for {aa}/{res3}")
            if "N" not in atom_names or "O" not in atom_names:
                raise ValueError(f"Backbone N/O missing for {aa}/{res3}")
            n_idx = atom_names.index("N")
            o_idx = atom_names.index("O")
            missing = 14 - n_real
            expected = Atom14SeqSync._BOLTZ_CODEBOOK[aa].n_markers
            if missing != expected:
                raise ValueError(
                    f"Boltz codebook mismatch for {aa}: missing={missing}, codebook_markers={expected}"
                )
            out[aa] = {"n_idx": n_idx, "o_idx": o_idx, "n_real": n_real}
        return out

    @staticmethod
    def _ensure_l14x3(t: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates to [L,14,3]."""
        if t.ndim == 4:
            if t.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 for coords, got {tuple(t.shape)}")
            t = t[0]
        if t.ndim != 3 or t.shape[1] != 14 or t.shape[2] != 3:
            raise ValueError(f"Expected [L,14,3], got {tuple(t.shape)}")
        return t

    @staticmethod
    def _ensure_l14(t: torch.Tensor) -> torch.Tensor:
        """Normalize mask to [L,14]."""
        if t.ndim == 3:
            if t.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 for mask, got {tuple(t.shape)}")
            t = t[0]
        if t.ndim != 2 or t.shape[1] != 14:
            raise ValueError(f"Expected [L,14], got {tuple(t.shape)}")
        return t

    def build_supervision(
        self,
        seq: str,
        cord_n14_tf: torch.Tensor,
        cmsk_n14_tf: torch.Tensor,
        cdr_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cord = self._ensure_l14x3(cord_n14_tf).clone()
        cmsk = self._ensure_l14(cmsk_n14_tf).clone().to(torch.bool)
        cdr_mask = cdr_mask.to(torch.bool).view(-1)

        if len(seq) != cord.shape[0]:
            raise ValueError(
                f"Sequence/coord length mismatch: len(seq)={len(seq)} "
                f"vs L={cord.shape[0]}"
            )

        marker_class = torch.full(
            (cord.shape[0], 14),
            -100,
            dtype=torch.long,
            device=cord.device,
        )
        marker_count_target = torch.zeros(
            (cord.shape[0], 2),
            dtype=cord.dtype,
            device=cord.device,
        )
        # Per-residue type label (codebook / RESD_NAMES_1C index) for the RAMF
        # decodability margin loss; -100 = ignore (non-CDR / unknown residue).
        # Indexing matches IDEAL_ATOM14_COORDS[i] <-> RESD_NAMES_1C[i].
        type_target = torch.full(
            (cord.shape[0],), -100, dtype=torch.long, device=cord.device
        )
        aa_to_type_idx = {aa: i for i, aa in enumerate(RESD_NAMES_1C)}

        n_idx = 0
        o_idx = 3

        for ridx, aa in enumerate(seq):
            if (
                ridx >= cdr_mask.numel()
                or not bool(cdr_mask[ridx])
                or aa not in self._n_real_dict
            ):
                continue

            if aa in aa_to_type_idx:
                type_target[ridx] = aa_to_type_idx[aa]

            n_real = self._n_real_dict[aa]
            code = self._BOLTZ_CODEBOOK[aa]

            marker_class[ridx, :n_real] = 0
            marker_count_target[ridx, 0] = float(code.n_on_n)
            marker_count_target[ridx, 1] = float(code.n_on_o)

            missing_slots = list(range(n_real, 14))
            marker_types = [1] * code.n_on_n + [2] * code.n_on_o
            marker_anchors = [n_idx] * code.n_on_n + [o_idx] * code.n_on_o

            if not (
                len(missing_slots)
                == len(marker_types)
                == len(marker_anchors)
            ):
                raise RuntimeError(
                    f"Marker count mismatch for residue {aa} at idx={ridx}"
                )

            for slot, marker_type, anchor_idx in zip(
                missing_slots,
                marker_types,
                marker_anchors,
            ):
                cord[ridx, slot] = cord[ridx, anchor_idx]
                cmsk[ridx, slot] = True
                marker_class[ridx, slot] = marker_type

        return {
            "cords_atom14": cord,
            "cmsk_atom14": cmsk.to(dtype=cmsk_n14_tf.dtype),
            "atom14_marker_class": marker_class,
            "atom14_marker_count_target": marker_count_target,
            "atom14_type_target": type_target,
        }

    def _count_no_markers(self, residue_atoms: torch.Tensor, residue_mask: torch.Tensor) -> Tuple[int, int]:
        """
        Count marker atoms within `self.decode_threshold` (1.0 A) of the fixed
        N (idx 0) or O (idx 3) anchor.  Relies on the fixed atom14 layout: slots
        0-3 are always the backbone N/CA/C/O, so markers can only ever land in
        slots 4-13 (min n_real over the codebook is 4).

        Slots 0-3 are ALL skipped, not just the two anchors.  Skipping only N/O
        left the real CA and C in the scan, and their distances to their own
        anchors are 1.46 A (CA-N) and 1.23 A (C-O) -- only 0.46 / 0.23 A of
        margin under a 1.0 A threshold.  A predicted C=O bond compressed below
        1.0 A therefore made the real C count as an O-marker and flipped the
        decoded residue type, which is why loss_bond was implicitly guarding the
        type channel.  Because n_real >= 4 for every residue, "slot < 4 is real
        backbone" needs no knowledge of the residue type, so this skip is NOT an
        n_real leak -- exactly the same standing as skipping N/O was.  Nearest
        remaining real atom is CB at ~2.4 A, so the margin is now ~1.4 A.
        """
        n_idx, o_idx = 0, 3
        n_pos = residue_atoms[n_idx]
        o_pos = residue_atoms[o_idx]
        n_count, o_count = 0, 0

        # Scan the marker-eligible slots only (4-13).
        for atom_idx in range(14):
            # Skip the four real backbone slots, and unmasked padded slots.
            if atom_idx < 4 or not bool(residue_mask[atom_idx]):
                continue

            atom = residue_atoms[atom_idx]
            d_n = torch.norm(atom - n_pos)
            d_o = torch.norm(atom - o_pos)

            # Farther than the threshold from both anchors => a real side-chain
            # atom, not a marker.
            if float(min(d_n.item(), d_o.item())) > self.decode_threshold:
                continue

            if d_n <= d_o:
                n_count += 1
            else:
                o_count += 1

        return n_count, o_count

    def decode_cdr_sequence(
            self,
            seq_true: str,
            pred_cord_n14_tf: torch.Tensor,
            pred_cmsk_n14_tf: torch.Tensor,
            cdr_mask: torch.Tensor,
    ) -> str:
        """Decode CDR residues by BoltzGen (#N,#O) marker counting; keep FR as native sequence."""
        cord = self._ensure_l14x3(pred_cord_n14_tf)
        cmsk = self._ensure_l14(pred_cmsk_n14_tf).to(torch.bool)
        cdr_mask = cdr_mask.to(torch.bool).view(-1)

        if len(seq_true) != cord.shape[0]:
            raise ValueError(f"Sequence/coord length mismatch: len(seq)={len(seq_true)} vs L={cord.shape[0]}")

        aa_list: List[str] = list(self._BOLTZ_CODEBOOK.keys())
        code_mat = torch.tensor(
            [[self._BOLTZ_CODEBOOK[aa].n_on_n, self._BOLTZ_CODEBOOK[aa].n_on_o] for aa in aa_list],
            dtype=cord.dtype,
            device=cord.device,
        )
        # Initialize with ground truth sequence to perfectly preserve FR regions
        seq_chars = list(seq_true)
        for ridx in range(len(seq_chars)):
            # Skip non-CDR residues (keep native FR sequence)
            if ridx >= cdr_mask.numel() or not bool(cdr_mask[ridx]):
                continue

            # Calculate N/O markers directly from coords without relying on seq_true[ridx]
            obs_n, obs_o = self._count_no_markers(cord[ridx], cmsk[ridx])
            obs = torch.tensor([obs_n, obs_o], dtype=cord.dtype, device=cord.device)
            dist = torch.sum(torch.abs(code_mat - obs.view(1, 2)), dim=-1)
            seq_chars[ridx] = aa_list[int(torch.argmin(dist).item())]

        return "".join(seq_chars)
