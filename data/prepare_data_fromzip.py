#!/usr/bin/env python3
"""Prepare training samples into torch serialized records."""

from __future__ import annotations

import zipfile
from functools import lru_cache
import argparse
import csv
import json
import pickle
import random
import shutil
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency in lightweight envs
    torch = None

from IgGM.protein.antibody_regions import (
    build_antibody_region_metadata,
    metadata_to_serializable,
    validate_antibody_region_metadata,
)

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
    "XLE": "J",
    "UNK": "X",
}

CHOTHIA_RANGES = {
    "cdr_L1": (24, 34),
    "cdr_L2": (50, 56),
    "cdr_L3": (89, 97),
    "cdr_H1": (26, 32),
    "cdr_H2": (52, 56),
    "cdr_H3": (95, 102),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset samples from SAbDab or local mock data")
    parser.add_argument("--raw_root", type=Path, default=Path("data/raw"), help="Root dir for raw SAbDab-like data")
    parser.add_argument("--out_root", type=Path, default=Path("data/processed"), help="Root output dir")
    parser.add_argument("--dataset_name", type=str, required=True, help="Output dataset folder name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--num_workers", type=int, default=8, help="Worker threads for sample conversion")
    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
        help="Reserved switch for downloading data (not required when local raw data exists)",
    )
    parser.add_argument("--build_splits", action="store_true", default=False, help="Build train/val/test prot_ids and train clusters")
    parser.add_argument("--split_out_dir", type=Path, default=None, help="Output directory for split files")
    parser.add_argument("--train_date_end", type=str, default="2022-12-31")
    parser.add_argument("--val_date_start", type=str, default="2023-01-01")
    parser.add_argument("--val_date_end", type=str, default="2023-06-30")
    parser.add_argument("--test_date_start", type=str, default="2023-07-01")
    parser.add_argument("--test_date_end", type=str, default="2023-12-30")
    parser.add_argument("--cluster_identity", type=float, default=0.95)
    return parser.parse_args()


def read_fasta_sequences(path: Path) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    current_header: Optional[str] = None
    chunks: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                sequences[current_header] = "".join(chunks)
            current_header = line[1:].strip() or f"seq_{len(sequences)}"
            chunks = []
        else:
            chunks.append(line)
    if current_header is not None:
        sequences[current_header] = "".join(chunks)
    return sequences


def find_sabdab_metadata(raw_root: Path) -> Optional[Path]:
    candidates: List[Path] = []
    patterns = ["*sabdab*.csv", "*sabdab*.tsv", "*metadata*.csv", "*metadata*.tsv"]
    for pattern in patterns:
        candidates.extend(raw_root.rglob(pattern))
    return sorted(set(candidates))[0] if candidates else None


def detect_mode(raw_root: Path) -> str:
    metadata = find_sabdab_metadata(raw_root)
    has_structure = any(raw_root.rglob("*.pdb")) or (find_structure_zip(raw_root) is not None)
    if metadata is not None and has_structure:
        return "real"
    return "mock"


def load_metadata_rows(metadata_path: Path) -> List[Dict[str, str]]:
    delimiter = "\t" if metadata_path.suffix.lower() == ".tsv" else ","
    with metadata_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter=delimiter)
        return [dict(row) for row in reader]

def find_structure_zip(raw_root: Path) -> Optional[Path]:
    zip_files = sorted(raw_root.glob("*.zip"))
    return zip_files[0] if zip_files else None
    
@lru_cache(maxsize=4096)
def _read_zip_member_lines(zip_path: str, member: str):
    """
    缓存 zip 内 pdb 文件内容，避免重复解压。
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member, "r") as fp:
            return tuple(
                line.decode("utf-8", errors="ignore")
                for line in fp
            )

def build_pdb_zip_index(zip_path: Path) -> Dict[str, Tuple[str, str]]:
    """
    返回:
        {
            pdb_stem_lower: (zip_path_str, inner_member_path)
        }
    仅索引 zip 内 all_structures/chothia/*.pdb
    """
    index: Dict[str, Tuple[str, str]] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            norm = name.replace("\\", "/")
            if not norm.startswith("all_structures/chothia/"):
                continue
            if not norm.endswith(".pdb"):
                continue
            stem = Path(norm).stem.lower()
            index.setdefault(stem, (str(zip_path), norm))
    return index


def parse_pdb_sequences_from_lines(lines: Iterable[str]) -> Dict[str, List[Tuple[int, str]]]:
    chain_residues: Dict[str, List[Tuple[int, str]]] = {}
    seen = set()
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        resname = line[17:20].strip().upper()
        chain_id = line[21].strip()
        if not chain_id:
            continue
        try:
            resseq = int(line[22:26].strip())
        except ValueError:
            continue
        icode = line[26].strip()
        residue_key = (chain_id, resseq, icode)
        if residue_key in seen:
            continue
        seen.add(residue_key)
        aa = AA3_TO_1.get(resname, "X")
        chain_residues.setdefault(chain_id, []).append((resseq, aa))
    return chain_residues

def read_pdb_lines_any(entry: Dict[str, object]) -> List[str]:
    zip_path = entry.get("pdb_zip_path")
    zip_member = entry.get("pdb_zip_member")

    if zip_path and zip_member:
        return list(_read_zip_member_lines(str(zip_path), str(zip_member)))

    pdb_path = Path(str(entry["pdb_path"]))
    return pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines()

def _rewrite_pdb_atom_line(line: str, atom_serial: int, new_chain_id: str, new_resseq: int) -> str:
    line = line.rstrip("\n")
    if len(line) < 80:
        line = line.ljust(80)

    # PDB fixed columns:
    #  1-6   record name
    #  7-11  atom serial
    # 22     chain id
    # 23-26  residue seq
    # 27     insertion code
    out = list(line)
    out[6:11] = f"{atom_serial:5d}"
    out[21] = new_chain_id
    out[22:26] = f"{new_resseq:4d}"
    out[26] = " "
    return "".join(out)

def write_processed_pdb(entry: Dict[str, object], pdb_dir: Path) -> Path:
    pdb_dir.mkdir(parents=True, exist_ok=True)

    chain_ids = parse_chain_ids(entry)
    file_stem = str(entry["file_stem"]) if "file_stem" in entry else build_sample_stem(entry)
    out_path = pdb_dir / f"{file_stem}.pdb"

    # 原始链 -> 新链名
    chain_plan: List[Tuple[str, str]] = []
    if chain_ids.get("H"):
        chain_plan.append((str(chain_ids["H"]), "H"))
    if chain_ids.get("L"):
        chain_plan.append((str(chain_ids["L"]), "L"))
    if chain_ids.get("A"):
        chain_plan.append((str(chain_ids["A"]), "A"))

    lines = read_pdb_lines_any(entry)

    out_lines: List[str] = []
    atom_serial = 1

    for old_chain, new_chain in chain_plan:
        residue_map: Dict[Tuple[int, str], int] = {}
        next_resseq = 1
        wrote_any = False

        for line in lines:
            if not line.startswith("ATOM"):
            # if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 27:
                continue

            chain_id = line[21].strip()
            if chain_id != old_chain:
                continue

            try:
                old_resseq = int(line[22:26].strip())
            except ValueError:
                continue
            icode = line[26].strip()

            residue_key = (old_resseq, icode)
            if residue_key not in residue_map:
                residue_map[residue_key] = next_resseq
                next_resseq += 1

            new_resseq = residue_map[residue_key]
            out_lines.append(_rewrite_pdb_atom_line(line, atom_serial, new_chain, new_resseq))
            atom_serial += 1
            wrote_any = True

        if wrote_any:
            out_lines.append(f"TER   {atom_serial:5d}")
            atom_serial += 1

    out_lines.append("END")
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return out_path

def parse_pdb_sequences_any(entry: Dict[str, object]) -> Dict[str, List[Tuple[int, str]]]:
    zip_path = entry.get("pdb_zip_path")
    zip_member = entry.get("pdb_zip_member")

    if zip_path and zip_member:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            with zf.open(str(zip_member), "r") as fp:
                lines = (line.decode("utf-8", errors="ignore") for line in fp)
                return parse_pdb_sequences_from_lines(lines)

    return parse_pdb_sequences(Path(str(entry["pdb_path"])))


def build_pdb_index(raw_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for pdb_file in raw_root.rglob("*.pdb"):
        index.setdefault(pdb_file.stem.lower(), pdb_file)
    return index


def choose_first(row: Dict[str, str], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value:
            return value.strip()
    return None


def parse_final_antigen_chain(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    parts = [x.strip() for x in text.split("|")]
    chain = parts[-1] if parts else None

    if chain in {"0", "NA", "None", "", None}:
        return None

    return chain

def is_supported_antigen_type(row: Dict[str, str]) -> bool:
    """Only keep samples with protein/peptide antigens."""
    antigen_type = choose_first(row, ["antigen_type", "antigen type", "antigenType"])
    if not antigen_type:
        return False
    val = str(antigen_type).strip().lower()
    return val in {"protein", "peptide"}

def parse_chain_from_sample_id(sample_id: str) -> Dict[str, Optional[str]]:
    parts = sample_id.split("_")
    chain_ids = {"H": None, "L": None, "A": None}
    if len(parts) >= 4:
        chain_ids["H"] = parts[1] if parts[1] != "NA" else None
        chain_ids["L"] = parts[2] if parts[2] != "NA" else None
        chain_ids["A"] = parts[3] if parts[3] != "NA" else None
    return chain_ids


def parse_chain_ids(entry: Dict[str, object]) -> Dict[str, Optional[str]]:
    row = entry.get("raw_row", {})
    chain_ids = parse_chain_from_sample_id(str(entry["sample_id"]))
    if isinstance(row, dict):
        chain_ids["H"] = chain_ids["H"] or choose_first(row, ["Hchain", "heavy_chain", "heavy", "chain_h"])
        chain_ids["L"] = chain_ids["L"] or choose_first(row, ["Lchain", "light_chain", "light", "chain_l"])
        # chain_ids["A"] = chain_ids["A"] or choose_first(
        #     row, ["antigen_chain", "antigen", "chain_a", "antigen_chain_id"]
        # )

        raw_antigen = choose_first(row, ["antigen_chain", "antigen", "chain_a", "antigen_chain_id"])
        final_antigen = parse_final_antigen_chain(raw_antigen)

        chain_ids["A"] = chain_ids["A"] or final_antigen

    return chain_ids

def _norm_chain_name(chain: Optional[str]) -> str:
    value = str(chain).strip() if chain is not None else ""
    return value if value and value != "0" else "NA"


def build_sample_stem(entry_or_sample: Dict[str, object]) -> str:
    chain_ids = parse_chain_ids(entry_or_sample)
    h = _norm_chain_name(chain_ids.get("H"))
    l = _norm_chain_name(chain_ids.get("L"))
    a = _norm_chain_name(chain_ids.get("A"))
    pdb_name = str(entry_or_sample["sample_id"])
    return f"{pdb_name}_{h}_{l}_{a}"

def parse_pdb_sequences(pdb_path: Path) -> Dict[str, List[Tuple[int, str]]]:
    chain_residues: Dict[str, List[Tuple[int, str]]] = {}
    seen = set()
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip().upper()
            chain_id = line[21].strip()
            if not chain_id:
                continue
            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                continue
            icode = line[26].strip()
            residue_key = (chain_id, resseq, icode)
            if residue_key in seen:
                continue
            seen.add(residue_key)
            aa = AA3_TO_1.get(resname, "X")
            chain_residues.setdefault(chain_id, []).append((resseq, aa))
    return chain_residues


def build_cdr_indices(chain_residues: List[Tuple[int, str]], cdr_names: List[str]) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    cdr_pdb: Dict[str, List[int]] = {}
    cdr_seq: Dict[str, List[int]] = {}
    for cdr_name in cdr_names:
        start, end = CHOTHIA_RANGES[cdr_name]
        pdb_positions: List[int] = []
        seq_positions: List[int] = []
        for seq_idx, (pdb_resseq, _) in enumerate(chain_residues, start=1):
            if start <= pdb_resseq <= end:
                pdb_positions.append(pdb_resseq)
                seq_positions.append(seq_idx)
        cdr_pdb[cdr_name] = pdb_positions
        cdr_seq[cdr_name] = seq_positions
    return cdr_pdb, cdr_seq


def extract_sequences_and_cdr(entry: Dict[str, object]) -> Tuple[Dict[str, str], Dict[str, List[int]], Dict[str, List[int]]]:
    chain_ids = parse_chain_ids(entry)
    parsed = parse_pdb_sequences_any(entry)

    sequences: Dict[str, str] = {}
    cdr_pdb: Dict[str, List[int]] = {}
    cdr_sequences: Dict[str, List[int]] = {}

    for role in ("H", "L", "A"):
        chain_id = chain_ids.get(role)
        if not chain_id or chain_id not in parsed:
            continue
        residues = parsed[chain_id]

        sequences[role] = "".join(aa for _, aa in residues)
        if role == "H":
            pdb_idx, seq_idx = build_cdr_indices(residues, ["cdr_H1", "cdr_H2", "cdr_H3"])
            cdr_pdb.update(pdb_idx)
            cdr_sequences.update(seq_idx)
        elif role == "L":
            pdb_idx, seq_idx = build_cdr_indices(residues, ["cdr_L1", "cdr_L2", "cdr_L3"])
            cdr_pdb.update(pdb_idx)
            cdr_sequences.update(seq_idx)

    return sequences, cdr_pdb, cdr_sequences


def write_processed_fasta(fasta_dir: Path, file_stem: str, sequences: Dict[str, str]) -> Path:
    fasta_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = fasta_dir / f"{file_stem}.fasta"
    lines: List[str] = []
    for tag in ("H", "L", "A"):
        seq = sequences.get(tag)
        if seq:
            lines.append(f">{tag}")
            lines.append(seq)
    fasta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fasta_path


def prepare_real_entries(raw_root: Path) -> List[Dict[str, object]]:
    metadata_path = find_sabdab_metadata(raw_root)
    if metadata_path is None:
        return []

    rows = load_metadata_rows(metadata_path)
    pdb_index = build_pdb_index(raw_root)
    structure_zip = find_structure_zip(raw_root)
    pdb_zip_index = build_pdb_zip_index(structure_zip) if structure_zip is not None else {}

    if structure_zip is not None:
        print(f"[INFO] Using structure zip: {structure_zip}, indexed pdb count: {len(pdb_zip_index)}")

    entries: List[Dict[str, object]] = []
    for row in rows:
        if not is_supported_antigen_type(row):
            continue
        antigen_raw = choose_first(row, ["antigen_chain", "antigen", "chain_a", "antigen_chain_id"])
        final_antigen = parse_final_antigen_chain(antigen_raw)
        if final_antigen is None:
            continue

        entry_id = choose_first(row, ["pdb", "pdb_id", "pdb_code", "id", "entry", "name"])
        if not entry_id:
            continue

        pdb_path_raw = choose_first(row, ["pdb_path", "structure_path", "pdb_file", "file"])
        pdb_path: Optional[Path] = None
        pdb_zip_path: Optional[str] = None
        pdb_zip_member: Optional[str] = None

        if pdb_path_raw:
            candidate = Path(pdb_path_raw)
            pdb_path = candidate if candidate.is_absolute() else (raw_root / candidate)
            if not pdb_path.exists():
                pdb_path = None

        if pdb_path is None:
            pdb_path = pdb_index.get(entry_id.lower())

        if pdb_path is None:
            zip_hit = pdb_zip_index.get(entry_id.lower())
            if zip_hit is not None:
                pdb_zip_path, pdb_zip_member = zip_hit

        if pdb_path is None and pdb_zip_member is None:
            continue

        entries.append(
            {
                "sample_id": entry_id,
                "mode": "real",
                "source_metadata": str(metadata_path),
                "pdb_path": str(pdb_path) if pdb_path is not None else "",
                "pdb_zip_path": pdb_zip_path,
                "pdb_zip_member": pdb_zip_member,
                "raw_row": row,
            }
        )
    return entries


def prepare_mock_entries() -> List[Dict[str, object]]:
    pdb_root = Path("examples/pdb.files.native")
    entries: List[Dict[str, object]] = []
    for pdb_path in sorted(pdb_root.glob("*.pdb")):
        entries.append(
            {
                "sample_id": pdb_path.stem,
                "mode": "mock",
                "pdb_path": str(pdb_path),
                "raw_row": {},
            }
        )
    return entries


def finalize_sample(entry: Dict[str, object], fasta_dir: Path) -> Dict[str, object]:
    sequences, cdr_pdb, cdr_sequences = extract_sequences_and_cdr(entry)
    if any("X" in str(seq) for seq in sequences.values()):
        # if exsit the ‘UNK’
        return None
    
    seq_lens = {name: len(seq) for name, seq in sequences.items()}
    file_stem = build_sample_stem(entry)
    fasta_path = write_processed_fasta(fasta_dir, file_stem, sequences)
    
    # extrct the ab regin meta
    region_metadata = build_antibody_region_metadata(
        sequence_lengths=seq_lens,
        cdr_sequences=cdr_sequences,
    )
    serialized_region = metadata_to_serializable(region_metadata)

    sample = {
        "sample_id": entry["sample_id"],
        "file_stem": file_stem,
        "mode": entry["mode"],
        "pdb_path": entry["pdb_path"],
        "pdb_zip_path": entry.get("pdb_zip_path"),
        "pdb_zip_member": entry.get("pdb_zip_member"),
        "fasta_path": str(fasta_path),
        "source_metadata": entry.get("source_metadata"),
        "sequences": sequences,
        "sequence_lengths": seq_lens,
        "cdr_pdb": cdr_pdb,
        "cdr_sequences": cdr_sequences,
        "antibody_region": serialized_region,
        "length_tensor": (
            torch.tensor(list(seq_lens.values()), dtype=torch.int32)
            if torch is not None
            else {"data": list(seq_lens.values()), "shape": [len(seq_lens)], "dtype": "int32" }
        ),
        "raw_row": entry.get("raw_row", {}),
    }
    return sample


def summarize_sample(sample: Dict[str, object]) -> Dict[str, object]:
    summary: Dict[str, object] = {"keys": sorted(sample.keys()), "tensors": {}, "sequences": {}}
    for key, value in sample.items():
        if torch is not None and torch.is_tensor(value):
            summary["tensors"][key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, dict) and {"shape", "dtype"}.issubset(value.keys()):
            summary["tensors"][key] = {"shape": value["shape"], "dtype": value["dtype"]}
    sequences = sample.get("sequences", {})
    if isinstance(sequences, dict):
        summary["sequences"] = {k: len(v) for k, v in sequences.items()}
    return summary


def _parse_date_safe(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    value = str(text).strip()
    fmts = ["%m/%d/%y", "%Y-%m-%d", "%m/%d/%Y"]
    for fmt in fmts:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _seq_identity(seq_a: str, seq_b: str) -> float:
    if not seq_a or not seq_b:
        return 0.0
    n = min(len(seq_a), len(seq_b))
    if n == 0:
        return 0.0
    matches = sum(1 for a, b in zip(seq_a[:n], seq_b[:n]) if a == b)
    return matches / n


def _greedy_cluster(ids: List[str], seqs: Dict[str, str], identity: float) -> List[List[str]]:
    clusters: List[List[str]] = []
    for prot_id in ids:
        assigned = False
        for cluster in clusters:
            rep = cluster[0]
            if _seq_identity(seqs.get(prot_id, ""), seqs.get(rep, "")) >= identity:
                cluster.append(prot_id)
                assigned = True
                break
        if not assigned:
            clusters.append([prot_id])
    return clusters


def _cluster_with_cdhit(ids: List[str], seqs: Dict[str, str], split_dir: Path, identity: float) -> List[List[str]]:
    # cdhit = shutil.which("cd-hit")
    # cdhit = shutil.which("/root/private_data/luog/codex/cd-hit-v4.8.1-2019-0228/cd-hit")
    cdhit = "/root/private_data/luog/codex/cd-hit-v4.8.1-2019-0228/cd-hit"
    if cdhit is None:
        return _greedy_cluster(ids, seqs, identity)

    fas = split_dir / "train_heavy.fasta"
    out = split_dir / "train_heavy_cdhit.fasta"
    clstr = Path(str(out) + ".clstr")
    with fas.open("w", encoding="utf-8") as handle:
        for prot_id in ids:
            handle.write(f">{prot_id}\n{seqs.get(prot_id, '')}\n")

    cmd = [cdhit, "-i", str(fas), "-o", str(out), "-c", f"{identity:.2f}", "-n", "5", "-d", "0"]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return _greedy_cluster(ids, seqs, identity)

    if not clstr.exists():
        return _greedy_cluster(ids, seqs, identity)

    clusters: List[List[str]] = []
    curr: List[str] = []
    for line in clstr.read_text(encoding="utf-8").splitlines():
        if line.startswith(">Cluster"):
            if curr:
                clusters.append(curr)
            curr = []
            continue
        if ">" in line:
            token = line.split(">", 1)[1].split("...", 1)[0]
            curr.append(token.strip())
    if curr:
        clusters.append(curr)
    return clusters if clusters else _greedy_cluster(ids, seqs, identity)


def build_paper_style_splits(entries: List[Dict[str, object]], args: argparse.Namespace, split_out_dir: Path) -> Dict[str, int]:
    split_out_dir.mkdir(parents=True, exist_ok=True)
    train_end = datetime.strptime(args.train_date_end, "%Y-%m-%d")
    val_start = datetime.strptime(args.val_date_start, "%Y-%m-%d")
    val_end = datetime.strptime(args.val_date_end, "%Y-%m-%d")
    test_start = datetime.strptime(args.test_date_start, "%Y-%m-%d")
    test_end = datetime.strptime(args.test_date_end, "%Y-%m-%d")

    train_ids: List[str] = []
    val_ids_raw: List[str] = []
    test_ids_raw: List[str] = []
    heavy_seq: Dict[str, str] = {}

    for entry in entries:
        row = entry.get("raw_row", {})
        if not isinstance(row, dict):
            continue
        dt = _parse_date_safe(row.get("date"))
        antigen_raw = choose_first(row, ["antigen_chain", "antigen", "chain_a", "antigen_chain_id"])
        final_antigen = parse_final_antigen_chain(antigen_raw)

        if final_antigen is None:
            continue

        if dt is None:
            continue
        chain_ids = parse_chain_ids(entry)
        heavy_chain = chain_ids.get("H")
        light_chain = chain_ids.get("L")
        ag_chain = chain_ids.get("A")
        if not heavy_chain or not ag_chain:
            continue
        if light_chain in {"0", "", None}:
            light_chain = None
        parsed = parse_pdb_sequences_any(entry)
        if heavy_chain not in parsed:
            continue
        h_seq = "".join(aa for _, aa in parsed[heavy_chain])
        if not h_seq:
            continue
        l_tok = light_chain if light_chain is not None else "NA"
        prot_id = f"{str(entry['sample_id']).lower()}_{heavy_chain}_{l_tok}_{ag_chain}"
        heavy_seq[prot_id] = h_seq
        if dt <= train_end:
            train_ids.append(prot_id)
        elif val_start <= dt <= val_end:
            val_ids_raw.append(prot_id)
        elif test_start <= dt <= test_end:
            test_ids_raw.append(prot_id)

    train_ids = sorted(set(train_ids))
    clusters = _cluster_with_cdhit(train_ids, heavy_seq, split_out_dir, args.cluster_identity)

    train_path = split_out_dir / "train_prot_ids.txt"
    train_path.write_text("\n".join(train_ids) + ("\n" if train_ids else ""), encoding="utf-8")

    cluster_path = split_out_dir / "train_prot_cluster.txt"
    with cluster_path.open("w", encoding="utf-8") as handle:
        for idx, cluster in enumerate(clusters):
            handle.write(f"cluster_{idx}\t" + " ".join(cluster) + "\n")

    def _dedup(ids: List[str], train_ids_all: List[str]) -> List[str]:
        out: List[str] = []
        train_seq = [heavy_seq[x] for x in train_ids_all if x in heavy_seq]
        for pid in sorted(set(ids)):
            seq = heavy_seq.get(pid, "")
            if not seq:
                continue
            if any(_seq_identity(seq, t) >= args.cluster_identity for t in train_seq):
                continue
            out.append(pid)
        return out

    val_ids = _dedup(val_ids_raw, train_ids)
    test_ids = _dedup(test_ids_raw, train_ids)

    (split_out_dir / "val_prot_ids.txt").write_text("\n".join(val_ids) + ("\n" if val_ids else ""), encoding="utf-8")
    (split_out_dir / "test_prot_ids.txt").write_text("\n".join(test_ids) + ("\n" if test_ids else ""), encoding="utf-8")

    return {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids), "clusters": len(clusters)}


def main() -> None:
    args = parse_args()

    if args.download:
        print("[INFO] --download was provided; automatic download is not implemented, using local files only.")

    mode = detect_mode(args.raw_root)
    if mode == "real":
        entries = prepare_real_entries(args.raw_root)
    else:
        entries = prepare_mock_entries()

    if not entries:
        raise RuntimeError("No entries found in either real mode or mock mode inputs.")

    rng = random.Random(args.seed)
    rng.shuffle(entries)
    if args.limit is not None and args.limit >= 0:
        entries = entries[: args.limit]

    dataset_root = args.out_root / args.dataset_name
    samples_dir = dataset_root / "samples"
    fasta_dir = args.out_root / "fasta"
    pdb_dir = args.out_root / "pdb"
    samples_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.num_workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        samples = list(
            tqdm(
                pool.map(lambda item: finalize_sample(item, fasta_dir), entries),
                total=len(entries),
                desc="Processing samples",
            )
        )
    
    samples = [s for s in samples if s is not None]
    for sample in tqdm(samples, total=len(samples), desc="Saving samples"):

        sample_path = samples_dir / f"{sample['file_stem']}.pt"
        processed_pdb_path = write_processed_pdb(sample, pdb_dir)
        sample["processed_pdb_path"] = str(processed_pdb_path)

        if torch is not None:
            torch.save(sample, sample_path)
        else:
            with sample_path.open("wb") as fp:
                pickle.dump(sample, fp)

    metadata = {
        "dataset_name": args.dataset_name,
        "mode": mode,
        "sample_count": len(samples),
        "samples_dir": str(samples_dir),
        "fasta_dir": str(fasta_dir),
        "seed": args.seed,
        "limit": args.limit,
        "num_workers": workers,
        "download": bool(args.download),
        "sample_ids": [s["sample_id"] for s in samples],
    }
    metadata_path = dataset_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.build_splits:
        split_out_dir = args.split_out_dir or dataset_root / "split"
        split_stat = build_paper_style_splits(entries, args, split_out_dir)
        print(f"Split files saved to: {split_out_dir}")
        print(json.dumps(split_stat, ensure_ascii=False, indent=2))

    first_summary = summarize_sample(samples[0])
    print(f"Sample count: {len(samples)}")
    print(f"Output path: {dataset_root}")
    print("First sample summary:")
    print(json.dumps(first_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# python data/prepare_data_fromzip.py \
#   --dataset_name sabdab \
#   --raw_root ./data/origin_file \
#   --out_root ./data/sabdab/processed \
#   --limit 20 \
#   --build_splits \
#   --split_out_dir ./data/sabdab/processed/sabdab_file/split \
#   --cluster_identity 0.95

# 代表性测试结果，7mi3文件

# python data/prepare_data_fromzip.py \
#   --dataset_name sabdab \
#   --raw_root ./data/origin_file \
#   --out_root ./data/sabdab \
#   --build_splits \
#   --split_out_dir ./data/sabdab/sabdab_file/split \
#   --cluster_identity 0.95 

# 测试新data

# python data/prepare_data_fromzip.py \
#   --dataset_name sabdab_2603 \
#   --raw_root ./data/origin_file \
#   --out_root ./data/sabdab_2603/processed \
#   --limit 100 \
#   --build_splits \
#   --split_out_dir ./data/sabdab_2603/processed/split \
#   --cluster_identity 0.95