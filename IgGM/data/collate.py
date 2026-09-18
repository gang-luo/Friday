"""Minimal collate function for processed SAbDab samples."""

from __future__ import annotations

from typing import Dict, List

import torch


def _pad_tensor_list(tensors: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    max_len = max(x.shape[0] for x in tensors)
    padded = []
    for x in tensors:
        if x.shape[0] == max_len:
            padded.append(x)
            continue
        pad_shape = (max_len - x.shape[0],) + x.shape[1:]
        pad = torch.full(pad_shape, pad_value, dtype=x.dtype)
        padded.append(torch.cat([x, pad], dim=0))
    return torch.stack(padded, dim=0)


def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Pad variable-length complex tensors and return a batched dictionary."""

    complex_items = [item["complex"] for item in batch]
    seqs = [item["seq"] for item in complex_items]
    lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long)

    cord = _pad_tensor_list([item["cord"] for item in complex_items], pad_value=0.0)
    cmsk = _pad_tensor_list([item["cmsk"] for item in complex_items], pad_value=0.0)
    asym_id = _pad_tensor_list([item["asym_id"].squeeze(0) for item in complex_items], pad_value=0).long()
    mask_ab = _pad_tensor_list([item["mask_ab"] for item in complex_items], pad_value=0).to(torch.int8)
    mask_design = _pad_tensor_list([item["mask_design"] for item in complex_items], pad_value=0).to(torch.int8)
    seq_mask = _pad_tensor_list([torch.ones(len(x), dtype=torch.int8) for x in seqs], pad_value=0).to(torch.int8)

    return {
        "sample_id": [item["sample_id"] for item in batch],
        "seq": seqs,
        "lengths": lengths,
        "cord": cord,
        "cmsk": cmsk,
        "asym_id": asym_id,
        "mask_ab": mask_ab,
        "mask_design": mask_design,
        "seq_mask": seq_mask,
        "raw": batch,
    }
