"""Structure helpers for loading a single chain from a PDB file."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from IgGM.protein.parser.pdb_parser import PdbParser


def load_chain_structure(path: str | Path, chain_id: str, aa_seq: Optional[str] = None) -> Dict[str, object]:
    """Load one chain as a dict with seq/cord/cmsk.

    Args:
        path: PDB path.
        chain_id: chain identifier.
        aa_seq: optional reference sequence.
    """

    seq, cord, cmsk, _, error = PdbParser.load(str(path), aa_seq=aa_seq, chain_id=chain_id)
    if error is not None:
        raise ValueError(f"Failed to parse chain {chain_id} from {path}: {error}")

    return {"seq": seq, "cord": cord, "cmsk": cmsk}
