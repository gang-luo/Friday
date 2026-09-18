from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch


LOOP_NAME_TO_ID = {
    "H1": 0,
    "H2": 1,
    "H3": 2,
    "L1": 3,
    "L2": 4,
    "L3": 5,
}

DEFAULT_LOOP_ORDER = ("H1", "H2", "H3", "L1", "L2", "L3")


@dataclass(frozen=True)
class LoopIndexLookup:
    """Index conversion helper for one loop.

    Attributes:
        global_indices: Global residue indices for valid loop residues, shape [loop_true_len].
        local_to_global: Padded local->global mapping, shape [loop_lmax]. Invalid entries are -1.
        global_to_local: Mapping from global residue index to local index.
    """

    global_indices: List[int]
    local_to_global: List[int]
    global_to_local: Dict[int, int]


@dataclass(frozen=True)
class AntibodyRegionValidationResult:
    """Validation summary for antibody region metadata."""

    ok: bool
    errors: List[str]



def _to_long_tensor(values: Sequence[int]) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.long)



def _chain_offsets(sequence_lengths: Mapping[str, int]) -> Dict[str, int]:
    h_len = int(sequence_lengths.get("H", 0))
    l_len = int(sequence_lengths.get("L", 0))
    return {"H": 0, "L": h_len, "A": h_len + l_len}



def _expected_loop_names(sequence_lengths: Mapping[str, int]) -> List[str]:
    names = ["H1", "H2", "H3"]
    if int(sequence_lengths.get("L", 0)) > 0:
        names.extend(["L1", "L2", "L3"])
    return names



def resolve_loop_lmax(
    loop_name: str,
    true_len: int,
    lmax_overrides: Optional[Mapping[str, int]] = None,
    default_lmax: Optional[int] = None,
) -> int:
    """Resolve per-loop Lmax.

    Priority: explicit loop override > explicit default_lmax > true_len.
    The returned value is always >= true_len and >= 1.
    """

    base = int(true_len)
    if lmax_overrides and loop_name in lmax_overrides:
        base = max(base, int(lmax_overrides[loop_name]))
    elif default_lmax is not None:
        base = max(base, int(default_lmax))
    return max(base, 1)



def _full_length(sequence_lengths: Mapping[str, int]) -> int:
    return int(sequence_lengths.get("H", 0)) + int(sequence_lengths.get("L", 0)) + int(sequence_lengths.get("A", 0))



def build_antibody_region_metadata(
    *,
    sequence_lengths: Mapping[str, int],
    cdr_sequences: Mapping[str, Sequence[int]],
    atom_mask: Optional[torch.Tensor] = None,
    lmax_overrides: Optional[Mapping[str, int]] = None,
    default_lmax: Optional[int] = None,
) -> Dict[str, object]:
    """Build antibody/FR/CDR region metadata.

    Args:
        sequence_lengths: Chain lengths with keys H/L/A.
        cdr_sequences: 1-based chain-local residue indices for cdr_H1 ... cdr_L3.
        atom_mask: Optional full-complex atom validity mask, shape [L, n_atom].
        lmax_overrides: Optional per-loop Lmax map.
        default_lmax: Optional fallback Lmax when no per-loop override is provided.

    Returns:
        Dictionary with residue-level, loop-level, and occupancy metadata.
        Tensor shapes:
            antibody_mask: [L]
            antigen_mask: [L]
            cdr_mask: [L]
            fr_mask: [L]
            loop_masks: [n_loops, L]
            loop_type_ids: [n_loops]
            loop_left_anchor_idx: [n_loops]
            loop_right_anchor_idx: [n_loops]
            loop_true_len: [n_loops]
            loop_lmax: [n_loops]
            loop_occ_target: [n_loops, max_loop_lmax]
            loop_valid_res_mask: [n_loops, max_loop_lmax]
            loop_atom_valid_mask: [n_loops, max_loop_lmax, n_atom]
            loop_global_res_indices: [n_loops, max_loop_lmax]
    """

    total_len = _full_length(sequence_lengths)
    offsets = _chain_offsets(sequence_lengths)
    h_len = int(sequence_lengths.get("H", 0))
    l_len = int(sequence_lengths.get("L", 0))
    ab_len = h_len + l_len

    antibody_mask = torch.zeros(total_len, dtype=torch.bool)
    antibody_mask[:ab_len] = True
    antigen_mask = ~antibody_mask
    cdr_mask = torch.zeros(total_len, dtype=torch.bool)

    loop_names = _expected_loop_names(sequence_lengths)
    loop_global_indices: List[List[int]] = []
    loop_true_lens: List[int] = []
    loop_lmax: List[int] = []
    loop_left_anchor_idx: List[int] = []
    loop_right_anchor_idx: List[int] = []
    loop_masks: List[torch.Tensor] = []
    loop_type_ids: List[int] = []

    for loop_name in loop_names:
        chain_id = loop_name[0]
        cdr_key = f"cdr_{loop_name}"
        offset = offsets[chain_id]
        local_indices = sorted({int(x) - 1 for x in cdr_sequences.get(cdr_key, []) if int(x) > 0})
        global_indices = [offset + idx for idx in local_indices]

        loop_mask = torch.zeros(total_len, dtype=torch.bool)
        if global_indices:
            loop_mask[global_indices] = True
            cdr_mask[global_indices] = True

        true_len = len(global_indices)
        curr_lmax = resolve_loop_lmax(loop_name, true_len, lmax_overrides=lmax_overrides, default_lmax=default_lmax)
        left_anchor = global_indices[0] - 1 if global_indices else -1
        right_anchor = global_indices[-1] + 1 if global_indices else -1

        loop_masks.append(loop_mask)
        loop_type_ids.append(LOOP_NAME_TO_ID[loop_name])
        loop_global_indices.append(global_indices)
        loop_true_lens.append(true_len)
        loop_lmax.append(curr_lmax)
        loop_left_anchor_idx.append(left_anchor)
        loop_right_anchor_idx.append(right_anchor)

    fr_mask = antibody_mask & (~cdr_mask)
    n_loops = len(loop_names)
    max_loop_lmax = max(loop_lmax) if loop_lmax else 1
    n_atom = int(atom_mask.shape[-1]) if atom_mask is not None else 0

    loop_occ_target = torch.zeros((n_loops, max_loop_lmax), dtype=torch.bool)
    loop_valid_res_mask = torch.zeros((n_loops, max_loop_lmax), dtype=torch.bool)
    loop_atom_valid_mask = torch.zeros((n_loops, max_loop_lmax, n_atom), dtype=torch.bool)
    loop_global_res_idx = torch.full((n_loops, max_loop_lmax), -1, dtype=torch.long)

    for i, global_indices in enumerate(loop_global_indices):
        true_len = loop_true_lens[i]
        curr_lmax = loop_lmax[i]
        if true_len > 0:
            loop_occ_target[i, :true_len] = True
            loop_valid_res_mask[i, :true_len] = True
            loop_global_res_idx[i, :true_len] = _to_long_tensor(global_indices)
            if atom_mask is not None and n_atom > 0:
                loop_atom_valid_mask[i, :true_len] = atom_mask[_to_long_tensor(global_indices)]
        if curr_lmax < max_loop_lmax:
            loop_global_res_idx[i, curr_lmax:] = -1
            loop_occ_target[i, curr_lmax:] = False
            loop_valid_res_mask[i, curr_lmax:] = False
            if n_atom > 0:
                loop_atom_valid_mask[i, curr_lmax:] = False

    metadata = {
        "chain_offsets": {k: int(v) for k, v in offsets.items()},
        "loop_names": loop_names,
        "loop_name_to_id": dict(LOOP_NAME_TO_ID),
        "max_loop_lmax": int(max_loop_lmax),
        "antibody_mask": antibody_mask,
        "antigen_mask": antigen_mask,
        "cdr_mask": cdr_mask,
        "fr_mask": fr_mask,
        "loop_masks": torch.stack(loop_masks, dim=0) if loop_masks else torch.zeros((0, total_len), dtype=torch.bool),
        "loop_type_ids": _to_long_tensor(loop_type_ids),
        "loop_left_anchor_idx": _to_long_tensor(loop_left_anchor_idx),
        "loop_right_anchor_idx": _to_long_tensor(loop_right_anchor_idx),
        "loop_true_len": _to_long_tensor(loop_true_lens),
        "loop_lmax": _to_long_tensor(loop_lmax),
        "loop_occ_target": loop_occ_target,
        "loop_valid_res_mask": loop_valid_res_mask,
        "loop_atom_valid_mask": loop_atom_valid_mask,
        "loop_global_res_indices": loop_global_res_idx,
    }
    return metadata



def full_to_loop_local_index(loop_global_res_indices: torch.Tensor, loop_idx: int, global_res_idx: int) -> int:
    """Convert a full-complex residue index to loop-local index; return -1 when not part of the loop."""

    row = loop_global_res_indices[int(loop_idx)]
    matches = torch.nonzero(row == int(global_res_idx), as_tuple=False)
    return int(matches[0, 0].item()) if matches.numel() > 0 else -1



def loop_local_to_full_index(loop_global_res_indices: torch.Tensor, loop_idx: int, local_idx: int) -> int:
    """Convert a loop-local residue index to full-complex residue index; return -1 for invalid/padded entries."""

    row = loop_global_res_indices[int(loop_idx)]
    if local_idx < 0 or local_idx >= row.shape[0]:
        return -1
    return int(row[int(local_idx)].item())



def find_first_valid_local_index(loop_valid_res_mask: torch.Tensor, loop_idx: int) -> int:
    """Find the first valid local residue index within one loop."""

    row = loop_valid_res_mask[int(loop_idx)]
    matches = torch.nonzero(row, as_tuple=False)
    return int(matches[0, 0].item()) if matches.numel() > 0 else -1



def find_last_valid_local_index(loop_valid_res_mask: torch.Tensor, loop_idx: int) -> int:
    """Find the last valid local residue index within one loop."""

    row = loop_valid_res_mask[int(loop_idx)]
    matches = torch.nonzero(row, as_tuple=False)
    return int(matches[-1, 0].item()) if matches.numel() > 0 else -1



def build_loop_index_lookups(metadata: Mapping[str, object]) -> List[LoopIndexLookup]:
    """Build conversion lookups for each loop from metadata tensors."""

    loop_global = metadata["loop_global_res_indices"]
    loop_true_len = metadata["loop_true_len"]
    lookups: List[LoopIndexLookup] = []
    for idx in range(loop_global.shape[0]):
        true_len = int(loop_true_len[idx].item())
        valid_globals = [int(x) for x in loop_global[idx, :true_len].tolist()]
        local_to_global = [int(x) for x in loop_global[idx].tolist()]
        global_to_local = {g: i for i, g in enumerate(valid_globals)}
        lookups.append(
            LoopIndexLookup(
                global_indices=valid_globals,
                local_to_global=local_to_global,
                global_to_local=global_to_local,
            )
        )
    return lookups



def validate_anchor_consistency(metadata: Mapping[str, object]) -> List[str]:
    """Check whether anchors are valid FR residues immediately flanking each loop."""

    errors: List[str] = []
    fr_mask = metadata["fr_mask"]
    cdr_mask = metadata["cdr_mask"]
    loop_names = list(metadata["loop_names"])
    left = metadata["loop_left_anchor_idx"]
    right = metadata["loop_right_anchor_idx"]
    loop_global = metadata["loop_global_res_indices"]
    loop_true_len = metadata["loop_true_len"]
    antibody_mask = metadata["antibody_mask"]

    for idx, loop_name in enumerate(loop_names):
        true_len = int(loop_true_len[idx].item())
        if true_len == 0:
            errors.append(f"{loop_name}: empty loop_true_len")
            continue
        first_global = int(loop_global[idx, 0].item())
        last_global = int(loop_global[idx, true_len - 1].item())
        left_idx = int(left[idx].item())
        right_idx = int(right[idx].item())
        if left_idx != first_global - 1:
            errors.append(f"{loop_name}: left anchor {left_idx} does not flank first residue {first_global}")
        if right_idx != last_global + 1:
            errors.append(f"{loop_name}: right anchor {right_idx} does not flank last residue {last_global}")
        for anchor_idx, side in ((left_idx, "left"), (right_idx, "right")):
            if anchor_idx < 0 or anchor_idx >= antibody_mask.shape[0] or not bool(antibody_mask[anchor_idx]):
                errors.append(f"{loop_name}: {side} anchor {anchor_idx} outside antibody region")
                continue
            if bool(cdr_mask[anchor_idx]):
                errors.append(f"{loop_name}: {side} anchor {anchor_idx} overlaps CDR region")
            if not bool(fr_mask[anchor_idx]):
                errors.append(f"{loop_name}: {side} anchor {anchor_idx} not inside FR mask")
    return errors



def validate_antibody_region_metadata(metadata: Mapping[str, object]) -> AntibodyRegionValidationResult:
    """Run lightweight consistency checks for antibody region metadata."""

    errors: List[str] = []
    fr_mask = metadata["fr_mask"].to(torch.bool)
    cdr_mask = metadata["cdr_mask"].to(torch.bool)
    loop_masks = metadata["loop_masks"].to(torch.bool)
    loop_occ_target = metadata["loop_occ_target"].to(torch.bool)
    loop_valid_res_mask = metadata["loop_valid_res_mask"].to(torch.bool)
    loop_true_len = metadata["loop_true_len"].to(torch.long)
    loop_atom_valid_mask = metadata["loop_atom_valid_mask"].to(torch.bool)
    loop_lmax = metadata["loop_lmax"].to(torch.long)
    loop_names = list(metadata["loop_names"])

    if torch.any(fr_mask & cdr_mask):
        errors.append("FR/CDR masks overlap")

    for idx, loop_name in enumerate(loop_names):
        loop_mask = loop_masks[idx]
        if torch.any(loop_mask & (~cdr_mask)):
            errors.append(f"{loop_name}: loop mask is not fully contained in cdr_mask")

        occ = loop_occ_target[idx]
        true_len = int(loop_true_len[idx].item())
        curr_lmax = int(loop_lmax[idx].item())
        if curr_lmax > occ.shape[0]:
            errors.append(f"{loop_name}: loop_lmax {curr_lmax} exceeds padded max {occ.shape[0]}")
        if torch.any(occ[curr_lmax:]):
            errors.append(f"{loop_name}: occupancy has valid entries beyond loop_lmax")
        if true_len != int(occ.sum().item()):
            errors.append(f"{loop_name}: loop_true_len {true_len} mismatches occupancy sum {int(occ.sum().item())}")
        if true_len != int(loop_valid_res_mask[idx].sum().item()):
            errors.append(f"{loop_name}: loop_true_len {true_len} mismatches residue valid mask sum")
        if true_len > 0:
            diffs = occ[1:].to(torch.int8) - occ[:-1].to(torch.int8)
            if torch.any(diffs > 0):
                errors.append(f"{loop_name}: occupancy target is not prefix-monotone")

        if loop_atom_valid_mask.shape[-1] > 0 and true_len < loop_atom_valid_mask.shape[1]:
            if torch.any(loop_atom_valid_mask[idx, true_len:]):
                errors.append(f"{loop_name}: padded residues have non-zero atom_valid_mask")

    errors.extend(validate_anchor_consistency(metadata))
    return AntibodyRegionValidationResult(ok=(len(errors) == 0), errors=errors)



def metadata_to_serializable(metadata: Mapping[str, object]) -> Dict[str, object]:
    """Convert metadata tensors into torch-save / JSON-friendly Python containers."""

    serializable: Dict[str, object] = {}
    for key, value in metadata.items():
        if torch.is_tensor(value):
            serializable[key] = value.cpu()
        elif isinstance(value, dict):
            serializable[key] = dict(value)
        elif isinstance(value, (list, tuple)):
            serializable[key] = list(value)
        else:
            serializable[key] = value
    return serializable
