# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""Lightning DataModule for local processed SAbDab records."""

from __future__ import annotations

import json
import pickle
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Sampler

from IgGM.data.convert_to_example_format import convert_entry_to_sample
from IgGM.data.sabdab import load_sabdab_metadata
from IgGM.protein.antibody_regions import (
    build_antibody_region_metadata,
    full_to_loop_local_index,
    loop_local_to_full_index,
)
from IgGM.protein import crop_sequence_with_epitope
from IgGM.protein.data_transform.processing_multimer import get_asym_ids

from .atom14_sync import Atom14SeqSync


@dataclass
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


class _ProteinSampleDataset(Dataset):
    """Lazy dataset that resolves a sample payload only when indexed."""

    def __init__(
        self,
        items: List[Dict[str, object]],
        n_steps: int = 200,
        cache_size: int = 128,
        chunk_size: Optional[int] = None,
        max_antigen_len: Optional[int] = None,
        subset_name: Optional[str] = None,
    ):
        self.items = items
        self.n_steps = n_steps
        self._subset_name = subset_name
        self._cache_size = max(1, int(cache_size))
        self._sample_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()        
        self._chunk_size = None if chunk_size is None else max(1, int(chunk_size))
        self._max_antigen_len = None if max_antigen_len is None else max(1, int(max_antigen_len))
        self._warn_count = 0
        self._atom14_sync = Atom14SeqSync()


    def _warn_once(self, msg: str) -> None:
        if self._warn_count < 8:
            print(f"[DataModule][warn] {msg}")
            self._warn_count += 1

    def _apply_antigen_crop(self, converted: Dict[str, Any]) -> Dict[str, Any]:
        if self._max_antigen_len is None:
            return converted

        chains = converted.get("chains", [])
        if not chains:
            return converted
        antigen = chains[-1]
        ag_seq = antigen.get("sequence") or antigen.get("seq")
        if not ag_seq or len(ag_seq) <= self._max_antigen_len:
            return converted

        epitope = antigen.get("epitope")
        if epitope is None:
            epitope = torch.zeros(len(ag_seq), dtype=torch.int8)
        if not torch.is_tensor(epitope):
            epitope = torch.as_tensor(epitope)

        seq_new, cord_new, cmsk_new, epitope_new, _ = crop_sequence_with_epitope(
            ag_seq,
            antigen["cord"],
            antigen["cmsk"],
            epitope,
            max_len=self._max_antigen_len,
        )

        antigen["seq"] = seq_new
        antigen["sequence"] = seq_new
        antigen["cord"] = cord_new
        antigen["cmsk"] = cmsk_new
        antigen["epitope"] = epitope_new.to(torch.int8)

        sequences = [x["sequence"] for x in chains]
        concatenated_seq = "".join(sequences)
        concatenated_cord = torch.cat([x["cord"] for x in chains], dim=0)
        concatenated_cmsk = torch.cat([x["cmsk"] for x in chains], dim=0)
        asym_id = len(chains) - get_asym_ids(sequences)
        mask_ab = torch.zeros(len(concatenated_seq), dtype=torch.int8)
        mask_ab[:-len(antigen["sequence"])] = 1

        complex_data = converted["complex"]
        complex_data["seq"] = concatenated_seq
        complex_data["cord"] = concatenated_cord
        complex_data["cmsk"] = concatenated_cmsk
        complex_data["asym_id"] = asym_id.unsqueeze(0)
        complex_data["mask_ab"] = mask_ab
        complex_data["a-cord"] = antigen["cord"]
        complex_data["a-cmsk"] = antigen["cmsk"]
        complex_data["epitope"] = antigen["epitope"]
        return converted

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _load_pt_record(sample_path: str) -> Dict[str, Any]:
        if sample_path.endswith(".pt"):
            return torch.load(sample_path, map_location="cpu",weights_only=True)
        with Path(sample_path).open("rb") as fp:
            return pickle.load(fp)

    def _cache_put(self, key: str, value: Dict[str, Any]) -> None:
        self._sample_cache[key] = value
        self._sample_cache.move_to_end(key)
        while len(self._sample_cache) > self._cache_size:
            self._sample_cache.popitem(last=False)
        
    def _resolve_sample_payload(self, item: Dict[str, Any]) -> Dict[str, Any]:
        sample_path = item.get("sample_path")
        processed_pdb_path = item.get("processed_pdb_path")
        prot_id = str(item.get("prot_id", ""))
        cache_key = str(sample_path or processed_pdb_path or prot_id)
        if cache_key in self._sample_cache: 
            self._sample_cache.move_to_end(cache_key)
            return self._sample_cache[cache_key]

        record: Dict[str, Any] = {}
        if sample_path and Path(str(sample_path)).exists():
            record = self._load_pt_record(str(sample_path))

        if not processed_pdb_path:
            processed_pdb_path = record.get("processed_pdb_path")
        if not processed_pdb_path:
            raise RuntimeError(f"Missing processed PDB path for sample: {prot_id}")

        seqs = record.get("sequences", {})
        light_chain_id = "L" if isinstance(seqs, dict) and seqs.get("L") else None
        if light_chain_id is None:
            parts = prot_id.split("_")
            if len(parts) >= 3 and parts[2].upper() != "NA":
                light_chain_id = "L"
        seq_lengths = dict(record.get("sequence_lengths") or {})
        if not seq_lengths and isinstance(seqs, dict):
            seq_lengths = {k: len(v) for k, v in seqs.items() if isinstance(v, str)}

        entry = {
            "pdb_id": str(record.get("sample_id") or item.get("pdb_id") or prot_id.split("_")[0]).lower(),
            "heavy_chain_id": "H",
            "light_chain_id": light_chain_id,
            "antigen_chain_ids": ["A"],
        }
        converted = convert_entry_to_sample(entry, str(processed_pdb_path))
        if converted is None:
            raise RuntimeError(f"Failed to convert processed PDB sample: {processed_pdb_path}")
        converted = self._apply_antigen_crop(converted)
        region_metadata = self._build_region_metadata(record, converted)

        complex_data = converted["complex"]
        mask_design = region_metadata["cdr_mask"].clone().to(torch.int8)
        atom14_sup = self._atom14_sync.build_supervision(
            seq=complex_data["seq"],
            cord_n14_tf=complex_data["cord"],
            cmsk_n14_tf=complex_data["cmsk"],
            cdr_mask=region_metadata["cdr_mask"],
        )

        payload = {
            "seq_true": complex_data["seq"],
            "prot_data_curr": {
                "seq": complex_data["seq"],
                "cord": complex_data["cord"],
                "cmsk": complex_data["cmsk"],
                "cords_atom14": atom14_sup["cords_atom14"],
                "cmsk_atom14": atom14_sup["cmsk_atom14"],
                "atom14_marker_class": atom14_sup["atom14_marker_class"],
                "atom14_marker_count_target": atom14_sup[
                    "atom14_marker_count_target"
                ],
                "atom14_type_target": atom14_sup.get("atom14_type_target"),
                "mask_design": mask_design,
                "mask_ab": complex_data["mask_ab"],
                "asym_id": complex_data["asym_id"],
                "a-cord": complex_data["a-cord"],
                "a-cmsk": complex_data["a-cmsk"],
                "epitope": complex_data["epitope"],
                # Use epitope-only conditioning (3-dim [B,L,1]) instead of the full
                # cross-chain contact map (4-dim [B,L,L,1]).  The 4-dim path leaks
                # the ground-truth docking pose and paratope, contradicting the
                # "no preset pose" claim.  See docs/adr/0003.
                "contact": None,
                **self._region_metadata_for_model(region_metadata),
            },
            "cdr_sequences": (record.get("cdr_sequences") or {}),
            "sequence_lengths": seq_lengths,
            "antibody_region": region_metadata,
        }
        # payload = self._center_complex_payload(payload)
        payload = self._center_antigen_payload(payload)
        self._cache_put(cache_key, payload)
        return payload

    @staticmethod
    def _region_metadata_for_model(region_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten region metadata into model input keys without breaking old callers."""

        return {
            "antibody_mask": region_metadata["antibody_mask"],
            "antigen_mask": region_metadata["antigen_mask"],
            "cdr_mask": region_metadata["cdr_mask"],
            "fr_mask": region_metadata["fr_mask"],
            "loop_masks": region_metadata["loop_masks"],
            "loop_type_ids": region_metadata["loop_type_ids"],
            "loop_names": list(region_metadata["loop_names"]),
            "loop_left_anchor_idx": region_metadata["loop_left_anchor_idx"],
            "loop_right_anchor_idx": region_metadata["loop_right_anchor_idx"],
            "loop_true_len": region_metadata["loop_true_len"],
            "loop_lmax": region_metadata["loop_lmax"],
            "loop_occ_target": region_metadata["loop_occ_target"],
            "loop_valid_res_mask": region_metadata["loop_valid_res_mask"],
            "loop_atom_valid_mask": region_metadata["loop_atom_valid_mask"],
            "loop_global_res_indices": region_metadata["loop_global_res_indices"],
        }

    @staticmethod
    def _to_tensor_dict(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Convert saved region metadata containers back to tensors."""

        out: Dict[str, Any] = {}
        for key, value in meta.items():
            if torch.is_tensor(value):
                out[key] = value.clone()
            elif isinstance(value, list) and value and all(isinstance(x, str) for x in value):
                out[key] = list(value)
            elif isinstance(value, dict):
                out[key] = dict(value)
            elif isinstance(value, list):
                out[key] = torch.as_tensor(value)
            else:
                out[key] = value
        return out

    def _build_region_metadata(self, record: Dict[str, Any], converted: Dict[str, Any]) -> Dict[str, Any]:
        """Build or refresh antibody region metadata for one converted sample."""

        complex_data = converted["complex"]
        seq_lengths = dict(record.get("sequence_lengths") or {})
        if not seq_lengths:
            seqs = record.get("sequences") or {}
            seq_lengths = {k: len(v) for k, v in seqs.items() if isinstance(v, str)}
        seq_lengths['A'] = seq_lengths['A'] if seq_lengths['A'] < self._max_antigen_len else self._max_antigen_len
        saved = record.get("antibody_region")
        if isinstance(saved, dict) and saved:
            region_metadata = self._to_tensor_dict(saved)
            atom_mask = complex_data["cmsk"]
            region_metadata = build_antibody_region_metadata(
                sequence_lengths=seq_lengths,
                cdr_sequences=(record.get("cdr_sequences") or {}),
                atom_mask=atom_mask,
                lmax_overrides={
                    str(name): int(lmax)
                    for name, lmax in zip(region_metadata.get("loop_names", []), region_metadata.get("loop_lmax", []))
                } if region_metadata.get("loop_names") is not None and region_metadata.get("loop_lmax") is not None else None,
            )
        else:
            region_metadata = build_antibody_region_metadata(
                sequence_lengths=seq_lengths,
                cdr_sequences=(record.get("cdr_sequences") or {}),
                atom_mask=complex_data["cmsk"],
            )

        # lightweight index conversion helpers for downstream loop-local code paths
        region_metadata["full_to_loop_local_index"] = full_to_loop_local_index
        region_metadata["loop_local_to_full_index"] = loop_local_to_full_index
        
        return region_metadata

    @staticmethod
    def _build_contact_map(complex_data: Dict[str, Any], cutoff: float = 8.0) -> Optional[torch.Tensor]:
        """Build residue-level interface contact map from C-alpha distances."""
        cord = complex_data.get("cord")
        cmsk = complex_data.get("cmsk")
        asym_id = complex_data.get("asym_id")
        if cord is None or cmsk is None or asym_id is None:
            return None

        if asym_id.ndim == 2:
            asym_id = asym_id[0]

        ca = cord[:, 1]
        ca_mask = cmsk[:, 1].to(ca.dtype)
        pair_dist = torch.cdist(ca, ca)
        chain_cross = (asym_id.unsqueeze(0) != asym_id.unsqueeze(1))
        valid = (ca_mask.unsqueeze(0) * ca_mask.unsqueeze(1)).bool()
        contact = (pair_dist <= float(cutoff)) & chain_cross & valid
        return contact.to(torch.float32)
    
    @staticmethod
    def _center_complex_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Center the whole complex by subtracting one global centroid."""
        prot = payload.get("prot_data_curr")
        if prot is None:
            return payload

        cord = prot.get("cord")
        cmsk = prot.get("cmsk")

        # Use atom mask to compute one centroid for the whole complex.
        atom_mask = cmsk.to(dtype=cord.dtype)
        denom = atom_mask.sum().clamp(min=1.0)
        center = (cord * atom_mask.unsqueeze(-1)).sum(dim=(0, 1)) / denom

        # Shift complex-level coordinates
        prot["cord"] = cord - center.view(1, 1, 3)
        if "cords_atom14" in prot and torch.is_tensor(prot["cords_atom14"]):
            prot["cords_atom14"] = prot["cords_atom14"] - center.view(1, 1, 3)

        # Shift antigen coordinates if present
        if "a-cord" in prot and torch.is_tensor(prot["a-cord"]):
            prot["a-cord"] = prot["a-cord"] - center.view(1, 1, 3)

        # Shift chain-level coordinates if they still exist in payload
        chains = payload.get("chains")
        if isinstance(chains, list):
            for chain in chains:
                if isinstance(chain, dict) and torch.is_tensor(chain.get("cord")):
                    chain["cord"] = chain["cord"] - center.view(1, 1, 3)
        return payload

    @staticmethod
    def _center_antigen_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """【核心修改】将整个复合物平移，使得【抗原的质心】严格对齐到全局原点 [0, 0, 0]"""
        prot = payload.get("prot_data_curr")
        if prot is None:
            return payload

        cord = prot.get("cord")
        cmsk = prot.get("cmsk")
        mask_ab = prot.get("mask_ab")

        # 提取抗原 mask (mask_ab == 0 表示抗原)
        # 为了严谨，结合 cmsk 确保原子是有效的 (通常取 CA 原子, 即 index 1)
        ag_mask = (mask_ab == 0).unsqueeze(-1) & cmsk[:, 1:2].to(torch.bool)
        ag_mask_float = ag_mask.to(dtype=cord.dtype)

        denom = ag_mask_float.sum().clamp(min=1.0)
        # 仅计算【抗原】的质心
        ag_center = (cord[:, 1:2, :] * ag_mask_float.unsqueeze(-1)).sum(dim=(0, 1)) / denom

        # 将抗原质心作为中心点，整体平移复合物的所有坐标
        prot["cord"] = cord - ag_center.view(1, 1, 3)
        if "cords_atom14" in prot and torch.is_tensor(prot["cords_atom14"]):
            prot["cords_atom14"] = prot["cords_atom14"] - ag_center.view(1, 1, 3)

        # Shift antigen coordinates if present
        if "a-cord" in prot and torch.is_tensor(prot["a-cord"]):
            prot["a-cord"] = prot["a-cord"] - ag_center.view(1, 1, 3)

        # Shift clean CDR loop local coords back to absolute frame for correct supervision
        # (这步通常在下游做，但全局平移保证了所有基准都是抗原)
        
        return payload
    
    def __getitem__(self, index):
        
        n_items = len(self.items)
        for _ in range(10): # 重复尝试返回结果
            item = self.items[index % n_items]
            step = random.randint(1, self.n_steps)
            try:
                payload = self._resolve_sample_payload(item)
                out = {
                    "idx_step": step,
                    "prot_id": item["prot_id"],
                    "payload": payload,
                    "inputs_addi": None,
                    "chunk_size": self._chunk_size,
                }
                if self._subset_name:
                    out["test_group"] = self._subset_name
                return out
            except Exception as exc:
                self._warn_once(f"skip invalid sample prot_id={item.get('prot_id')} reason={exc}")
                index = random.randint(0, n_items - 1)

        item = self.items[index % n_items]
        raise RuntimeError(
            f"Failed to resolve sample after {10} retries; "
            f"last prot_id={item.get('prot_id')}"
        )


class ClusterEpochSampler(Sampler[int]):
    """Sample one item per cluster for each epoch."""

    def __init__(self, clusters: List[List[int]], seed: int = 42) -> None:
        self.clusters = [c for c in clusters if c]
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = [rng.choice(cluster) for cluster in self.clusters]
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.clusters)


def _batch_one(batch):
    return batch[0]


def _load_id_set(path: Optional[str | Path]) -> Optional[set[str]]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    ids = {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}
    return ids if ids else None


def _parse_prot_id(prot_id: str) -> Optional[Dict[str, object]]:
    """Parse split `prot_id` string to a minimal metadata-like entry."""
    parts = prot_id.strip().split("_")
    if len(parts) != 4:
        return None
    pdb_id, heavy_id, light_id, antigen_id = parts
    light = None if light_id.upper() == "NA" else light_id
    return {
        "pdb_id": pdb_id.lower(),
        "heavy_chain_id": heavy_id,
        "light_chain_id": light,
        "antigen_chain_ids": [antigen_id],
        "prot_id": prot_id,
        "file_stem": prot_id,
    }


def _is_nanobody_prot_id(prot_id: str) -> bool:
    """Classify nanobody by the light-chain field in `pdb_H_L/NA_A` prot_id."""
    parts = str(prot_id).strip().split("_")
    return len(parts) >= 3 and parts[2].upper() == "NA"


class ProcessedSabdabDataModule(pl.LightningDataModule):
    def __init__(
        self,
        metadata_path: str | Path,
        pdb_dir: str | Path,
        batch_size: int = 1,
        num_workers: int = 0,
        split_cfg: Optional[SplitConfig] = None,
        train_ids_path: Optional[str | Path] = None,
        val_ids_path: Optional[str | Path] = None,
        test_ids_path: Optional[str | Path] = None,
        test_nanobody_ids_path: Optional[str | Path] = None,
        test_standard_ids_path: Optional[str | Path] = None,
        train_cluster_path: Optional[str | Path] = None,
        samples_dir: Optional[str | Path] = None,
        n_steps: int = 200,
        lazy_cache_size: int = 128,
        forward_chunk_size: Optional[int] = None,
        max_antigen_len: Optional[int] = None,
    ):
        super().__init__()
        self.metadata_path = Path(metadata_path)
        self.pdb_dir = Path(pdb_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_cfg = split_cfg or SplitConfig()
        self.train_ids = _load_id_set(train_ids_path)
        self.val_ids = _load_id_set(val_ids_path)
        self.test_ids = _load_id_set(test_ids_path)
        self.test_nanobody_ids = _load_id_set(test_nanobody_ids_path)
        self.test_standard_ids = _load_id_set(test_standard_ids_path)
        self.train_cluster_path = Path(train_cluster_path) if train_cluster_path else None
        self.samples_dir = Path(samples_dir) if samples_dir else None
        self.n_steps = n_steps
        self.lazy_cache_size = lazy_cache_size
        self.forward_chunk_size = None if forward_chunk_size is None else max(1, int(forward_chunk_size))
        self.max_antigen_len = None if max_antigen_len is None else max(1, int(max_antigen_len))
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None
        self.test_datasets: OrderedDict[str, Dataset] = OrderedDict()
        self._train_sampler: Optional[Sampler[int]] = None
        self.is_persistent = self.num_workers > 0


    def _resolve_samples_dir(self, meta: Dict[str, object]) -> Optional[Path]:
        if self.samples_dir is not None:
            return self.samples_dir
        sample_dir = meta.get("samples_dir")
        if sample_dir:
            p = Path(str(sample_dir))
            if p.exists():
                return p
            p2 = self.metadata_path.parents[2] / p.relative_to("data") if str(p).startswith("data/") else None
            if p2 is not None and p2.exists():
                return p2
        candidate = self.metadata_path.parent / "samples"
        return candidate if candidate.exists() else None

    def _load_entries(self) -> List[Dict[str, object]]:
        meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        entries = meta.get("entries", [])
        if entries:
            return entries

        split_ids: set[str] = set()
        for id_set in (
            self.train_ids,
            self.val_ids,
            self.test_ids,
            self.test_nanobody_ids,
            self.test_standard_ids,
        ):
            if id_set:
                split_ids.update(id_set)
        if split_ids:
            parsed = [_parse_prot_id(x) for x in sorted(split_ids)]
            return [x for x in parsed if x is not None]

        sample_ids = meta.get("sample_ids", [])
        if sample_ids:
            stems = {Path(p).stem for p in self.pdb_dir.glob("*.pdb")}
            parsed = []
            for stem in sorted(stems):
                if not any(stem == sid or stem.startswith(f"{sid}_") for sid in sample_ids):
                    continue
                item = _parse_prot_id(stem)
                if item is not None:
                    parsed.append(item)
            if parsed:
                return parsed

        tsv_path = self.metadata_path.parents[2] / "metadata" / "sabdab.tsv"
        if tsv_path.exists():
            entries = load_sabdab_metadata(tsv_path)
            sample_ids = set(meta.get("sample_ids", []))
            if sample_ids:
                entries = [e for e in entries if e.get("pdb_id") in sample_ids]
        return entries

    def _build_items(self, entries: Iterable[Dict[str, object]], samples_dir: Optional[Path]) -> List[Dict[str, object]]:
        items = []
        sample_index: Dict[str, Path] = {}
        if samples_dir is not None and samples_dir.exists():
            sample_index = {p.stem: p for p in samples_dir.glob("*.pt")}

        for entry in entries:
            prot_id = str(entry.get("prot_id") or "").strip()
            if not prot_id:
                pdb_id = str(entry.get("pdb_id", "")).lower()
                heavy = str(entry.get("heavy_chain_id") or "").strip()
                light = str(entry.get("light_chain_id") or "NA").strip() or "NA"
                if light in {"0", "None", "none"}:
                    light = "NA"
                ag_list = entry.get("antigen_chain_ids") or []
                ag = str(ag_list[0]).strip() if ag_list else ""
                if not pdb_id or not heavy or not ag:
                    continue
                prot_id = f"{pdb_id}_{heavy}_{light}_{ag}"

            sample_path = sample_index.get(prot_id)
            processed_pdb_path = self.pdb_dir / f"{prot_id}.pdb"
            if not processed_pdb_path.exists():
                continue

            items.append(
                {
                    "prot_id": prot_id,
                    "pdb_id": str(entry.get("pdb_id", "")).lower(),
                    "sample_path": str(sample_path) if sample_path is not None else None,
                    "processed_pdb_path": str(processed_pdb_path),
                }
            )
        return items

    def _build_sampler_from_cluster_file(self, train_items: List[Dict[str, object]]) -> Optional[Sampler[int]]:
        if self.train_cluster_path is None or not self.train_cluster_path.exists():
            return None
        id2idx = {x["prot_id"]: i for i, x in enumerate(train_items)}
        clusters: List[List[int]] = []
        for line in self.train_cluster_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            members = parts[-1].split()
            idxs = [id2idx[m] for m in members if m in id2idx]
            if idxs:
                clusters.append(idxs)
        return ClusterEpochSampler(clusters, seed=self.split_cfg.seed) if clusters else None

    def _make_dataset(self, items: List[Dict[str, object]], subset_name: Optional[str] = None) -> Dataset:
        return _ProteinSampleDataset(
            items,
            n_steps=self.n_steps,
            cache_size=self.lazy_cache_size,
            chunk_size=self.forward_chunk_size,
            max_antigen_len=self.max_antigen_len,
            subset_name=subset_name,
        )

    def _make_eval_loader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.is_persistent,
            collate_fn=_batch_one,
        )

    def setup(self, stage: Optional[str] = None):
        meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        samples_dir = self._resolve_samples_dir(meta)
        entries = self._load_entries()
        items = self._build_items(entries, samples_dir)
        if not items:
            raise RuntimeError(f"No valid samples found from {self.metadata_path} and {self.pdb_dir}")

        test_id_union: set[str] = set()
        for id_set in (self.test_ids, self.test_nanobody_ids, self.test_standard_ids):
            if id_set:
                test_id_union.update(id_set)

        if self.train_ids is not None:
            train_items = [x for x in items if x["prot_id"] in self.train_ids]
            val_items = [x for x in items if self.val_ids and x["prot_id"] in self.val_ids]
            test_items = [x for x in items if test_id_union and x["prot_id"] in test_id_union]
        else:
            rnd = random.Random(self.split_cfg.seed)
            rnd.shuffle(items)
            n = len(items)
            n_train = max(1, int(n * self.split_cfg.train_ratio))
            n_val = max(1, int(n * self.split_cfg.val_ratio)) if n > 2 else 0
            train_items = items[:n_train]
            val_items = items[n_train:n_train + n_val]
            test_items = items[n_train + n_val:] if (n_train + n_val) < n else items[-1:]

        # Build type-specific test subsets.
        # If explicit files are provided, they take precedence; otherwise split the normal test_ids by prot_id.
        if self.test_nanobody_ids is not None:
            test_nanobody_items = [x for x in items if x["prot_id"] in self.test_nanobody_ids]
        else:
            test_nanobody_items = [x for x in test_items if _is_nanobody_prot_id(str(x["prot_id"]))]

        if self.test_standard_ids is not None:
            test_standard_items = [x for x in items if x["prot_id"] in self.test_standard_ids]
        else:
            test_standard_items = [x for x in test_items if not _is_nanobody_prot_id(str(x["prot_id"]))]

        self.train_ds = self._make_dataset(train_items)
        self.val_ds = self._make_dataset(val_items if val_items else train_items[:1])
        self.test_ds = self._make_dataset(test_items if test_items else train_items[:1])

        self.test_datasets = OrderedDict()
        if test_nanobody_items:
            self.test_datasets["nanobody"] = self._make_dataset(test_nanobody_items, subset_name="nanobody")
        if test_standard_items:
            self.test_datasets["standard"] = self._make_dataset(test_standard_items, subset_name="standard")
        if not self.test_datasets:
            self.test_datasets["all"] = self.test_ds

        print(
            "[DataModule] test split:",
            ", ".join(f"{name}={len(ds)}" for name, ds in self.test_datasets.items()),
        )

        self._train_sampler = self._build_sampler_from_cluster_file(train_items)

    def train_dataloader(self):
        if self._train_sampler is not None:
            return DataLoader(self.train_ds, batch_size=1, shuffle=False, 
                              sampler=self._train_sampler, num_workers=self.num_workers, persistent_workers=self.is_persistent, 
                              collate_fn=_batch_one)
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, persistent_workers=self.is_persistent, 
                          num_workers=self.num_workers, 
                          collate_fn=_batch_one)
    
    def val_dataloader(self):
        return self._make_eval_loader(self.val_ds)

    def test_dataloader(self):
        if self.test_datasets:
            loaders = [self._make_eval_loader(ds) for ds in self.test_datasets.values()]
            return loaders if len(loaders) > 1 else loaders[0]
        return self._make_eval_loader(self.test_ds)

