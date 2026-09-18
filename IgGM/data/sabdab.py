"""Utilities for loading and normalizing local SAbDab metadata files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional


_FIELD_CANDIDATES = {
    "pdb_id": ["pdb", "pdb_id", "pdbid"],
    "heavy_chain_id": ["hchain", "h_chain", "heavy_chain", "heavy_chain_id"],
    "light_chain_id": ["lchain", "l_chain", "light_chain", "light_chain_id"],
    "antigen_chain_id": [
        "antigen_chain",
        "antigen_chain_id",
        "ag_chain",
        "ag_chains",
        "antigen_chains",
    ],
    "release_date": ["date", "release_date", "deposition_date"],
    "resolution": ["resolution", "reso", "resolution_(\u00c5)"],
    "species": ["species", "organism", "host_species"],
    "antigen_type": ["antigen_type", "antigen", "antigen_type_detail"],
}


def _normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _normalize_chain_ids(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    text = str(value).replace("|", ",").replace(";", ",")
    parts = [x.strip() for x in text.split(",") if x.strip()]
    return parts


def _resolve_column(row: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in row and row[candidate] != "":
            return row[candidate]
    return None


def load_sabdab_metadata(path: str | Path, delimiter: Optional[str] = None) -> List[Dict[str, object]]:
    """Load a local CSV/TSV metadata file and normalize output fields.

    Returns dictionaries with normalized keys:
    - pdb_id
    - heavy_chain_id
    - light_chain_id
    - antigen_chain_ids
    - release_date
    - resolution
    - species
    - antigen_type
    - is_nanobody
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SAbDab metadata file not found: {path}")

    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","

    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows

        normalized_fieldnames = [_normalize_header(x) for x in reader.fieldnames]
        for raw_row in reader:
            row = {normalized_fieldnames[i]: (v.strip() if isinstance(v, str) else v) for i, v in enumerate(raw_row.values())}
            pdb_id = _resolve_column(row, _FIELD_CANDIDATES["pdb_id"])
            if not pdb_id:
                continue

            heavy_chain = _resolve_column(row, _FIELD_CANDIDATES["heavy_chain_id"])
            light_chain = _resolve_column(row, _FIELD_CANDIDATES["light_chain_id"])
            antigen_chain = _resolve_column(row, _FIELD_CANDIDATES["antigen_chain_id"])
            resolution = _resolve_column(row, _FIELD_CANDIDATES["resolution"])

            rows.append(
                {
                    "pdb_id": pdb_id.lower(),
                    "heavy_chain_id": heavy_chain,
                    "light_chain_id": light_chain,
                    "antigen_chain_ids": _normalize_chain_ids(antigen_chain),
                    "release_date": _resolve_column(row, _FIELD_CANDIDATES["release_date"]),
                    "resolution": float(resolution) if resolution not in (None, "") else None,
                    "species": _resolve_column(row, _FIELD_CANDIDATES["species"]),
                    "antigen_type": _resolve_column(row, _FIELD_CANDIDATES["antigen_type"]),
                    "is_nanobody": bool(heavy_chain) and not bool(light_chain),
                }
            )

    return rows
