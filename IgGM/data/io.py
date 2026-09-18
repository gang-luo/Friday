"""Input/output helpers for processed SAbDab samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch


def save_sample(path: str | Path, sample: Dict[str, object]) -> None:
    """Save one sample as `<path>/sample.pt` plus `<path>/metadata.json`."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    payload = dict(sample)
    metadata = payload.get("metadata", {})

    torch.save(payload, path / "sample.pt")
    with (path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def load_sample(path: str | Path) -> Dict[str, object]:
    """Load one sample from `<path>/sample.pt` and attach `metadata.json` if present."""

    path = Path(path)
    sample = torch.load(path / "sample.pt", map_location="cpu")
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            sample["metadata"] = json.load(handle)
    return sample
