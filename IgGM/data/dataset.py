"""Dataset for processed SAbDab samples saved on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset

from .io import load_sample


class ProcessedSabdabDataset(Dataset):
    """Dataset that loads one preprocessed sample dictionary per item."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.sample_dirs: List[Path] = sorted(
            [p for p in self.root.iterdir() if p.is_dir() and (p / "sample.pt").exists()]
        )

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return load_sample(self.sample_dirs[index])
