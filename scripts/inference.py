#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test-time inference for AbG antibody design.

Usage:
    python scripts/inference.py \\
        --fasta examples/1bvk.fasta \\
        --antigen examples/1bvk_antigen.pdb \\
        --fv examples/1bvk_fv.pdb \\
        --design_ckpt checkpoints/precise-epoch=02.ckpt \\
        --epitope 10,11,12,13,14 \\
        --output outputs/1bvk_design \\
        --num_samples 4 \\
        --steps 20 \\
        --seed 20260909

Inputs:
    - FASTA: H[/L]/A chains, X marks design positions (CDR loops)
    - Antigen PDB: provides receptor structure and antigen coordinates
    - Fv PDB: provides framework internal conformation (required input, not generated)
    - Epitope: comma-separated 1-based residue indices on the antigen chain

Outputs (per sample):
    - {output}_sample{i}.pdb: predicted complex structure
    - {output}_sample{i}.fasta: two sequences:
        > geo_decoded | CDR types decoded from atom14 geometry (reported metric)
        > seq_head    | sequence-head feedback chain (auxiliary, for comparison only)
    - {output}_metrics.json: structural metrics if --ref_pdb is given
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from IgGM.model import build_design_model_module, build_ppi_featurizer_module
from IgGM.model.arch.core.diffuser import Diffuser
from IgGM.protein import ProtStruct, build_antibody_region_metadata
from IgGM.protein.data_transform import get_asym_ids
from IgGM.protein.parser import PdbParser
from src.iggm_lightning import (
    Atom14SeqSync,
    StructureMetrics,
    MetricConfig,
    load_design_state_dict,
    run_reverse_sampling,
)


logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="AbG antibody design inference")

    # Required inputs
    parser.add_argument("--fasta", "-f", required=True, help="FASTA with H[/L]/A, X marks design")
    parser.add_argument("--antigen", "-ag", required=True, help="Antigen PDB file")
    parser.add_argument("--fv", required=True, help="Antibody Fv PDB (FR conformation source)")

    # Model
    parser.add_argument("--design_ckpt", required=True, help="Training checkpoint (.ckpt)")
    parser.add_argument(
        "--ppi_ckpt",
        default="/root/private_data/luog/codex/IgGM/checkpoints/esm_ppi_650m_ab.pth",
        help="PLM featurizer checkpoint",
    )
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA weights (default)")
    parser.add_argument("--no_ema", dest="use_ema", action="store_false", help="Use raw weights")

    # Conditioning
    parser.add_argument(
        "--epitope",
        help="Comma-separated 1-based antigen residue indices, e.g. '10,11,12,13,14'",
    )
    parser.add_argument(
        "--cal_epitope",
        type=float,
        metavar="DIST",
        help="Auto-compute epitope: antigen residues within DIST Å of Fv",
    )

    # Sampling
    parser.add_argument("--steps", type=int, default=20, help="Reverse sampling steps (thinning)")
    parser.add_argument("--num_samples", "-ns", type=int, default=1, help="Samples per input")
    parser.add_argument("--seed", type=int, default=20260909, help="Base seed (sample i uses seed+i)")
    parser.add_argument(
        "--seq_feedback",
        action="store_true",
        default=True,
        help="Close sequence loop (feed back seq-head prediction, default on)",
    )
    parser.add_argument(
        "--no_seq_feedback",
        dest="seq_feedback",
        action="store_false",
        help="Open sequence loop (re-noise ground truth, A/B control only)",
    )

    # Design scope
    parser.add_argument(
        "--design_loops",
        help="Explicitly name which loops X spans map to, e.g. 'H3' or 'H1,H2,H3'",
    )

    # Output
    parser.add_argument("--output", "-o", required=True, help="Output path prefix")
    parser.add_argument("--ref_pdb", help="Reference PDB for metrics (optional)")

    # Device
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if not args.epitope and not args.cal_epitope:
        parser.error("Must provide --epitope or --cal_epitope")
    if args.epitope and args.cal_epitope:
        parser.error("Cannot specify both --epitope and --cal_epitope")

    return args


def load_models(args) -> Tuple[nn.Module, nn.Module, Diffuser]:
    """Load design model, PLM featurizer, and diffuser."""
    device = torch.device(args.device)

    logger.info(f"Loading PLM featurizer from {args.ppi_ckpt}")
    plm_featurizer = build_ppi_featurizer_module(args.ppi_ckpt).to(device).eval()

    logger.info(f"Loading design checkpoint from {args.design_ckpt}")
    state_dict, which = load_design_state_dict(args.design_ckpt, use_ema=args.use_ema)
    logger.info(f"Loaded {which} weights")

    # Build model from state_dict keys (config is baked into the architecture)
    from IgGM.config import get_cfg_defaults
    config = get_cfg_defaults()
    design_model = build_design_model_module(None, config).to(device).eval()
    design_model.load_state_dict(state_dict, strict=False)

    diffuser = Diffuser(n_steps=200)

    return design_model, plm_featurizer, diffuser


def build_prot_data_curr(args) -> Dict[str, Any]:
    """Build prot_data_curr from FASTA + antigen PDB + Fv PDB.

    This is the inference equivalent of _ProteinSampleDataset._resolve_sample_payload.
    The key difference: CDR coordinates are ZEROED and masks set to all-14, so the
    network input is strictly leak-free (see docs/adr/0003).
    """
    device = torch.device(args.device)

    # Parse FASTA
    from IgGM.protein.parser import parse_fasta
    fasta_records = parse_fasta(args.fasta)
    if len(fasta_records) not in (2, 3):
        raise ValueError(f"FASTA must have 2 (H/A) or 3 (H/L/A) chains, got {len(fasta_records)}")

    has_light = len(fasta_records) == 3
    if has_light:
        h_seq, l_seq, ag_seq = [r["seq"] for r in fasta_records]
        ab_seq = h_seq + l_seq
        chain_ids_fv = ["H", "L"]
        chain_id_ag = "A" if len(set("HLA")) == 3 else fasta_records[-1]["id"]
    else:
        h_seq, ag_seq = [r["seq"] for r in fasta_records]
        l_seq = ""
        ab_seq = h_seq
        chain_ids_fv = ["H"]
        chain_id_ag = fasta_records[-1]["id"]

    # Load Fv structure (provides FR internal conformation)
    logger.info(f"Loading Fv from {args.fv}")
    fv_coords_list, fv_masks_list = [], []
    for ch in chain_ids_fv:
        seq_pdb, cord, cmsk, _, err = PdbParser.load(args.fv, chain_id=ch)
        if err:
            raise RuntimeError(f"Failed to load chain {ch} from {args.fv}: {err}")
        # Verify sequence length matches FASTA
        expected = h_seq if ch == "H" else l_seq
        if len(seq_pdb) != len(expected):
            raise ValueError(
                f"Fv chain {ch} length mismatch: PDB has {len(seq_pdb)} residues, "
                f"FASTA has {len(expected)}"
            )
        fv_coords_list.append(cord)
        fv_masks_list.append(cmsk)

    fv_coords = torch.cat(fv_coords_list, dim=0)  # [ab_len, 14, 3]
    fv_masks = torch.cat(fv_masks_list, dim=0)    # [ab_len, 14]

    # Load antigen structure
    logger.info(f"Loading antigen from {args.antigen}")
    ag_seq_pdb, ag_coords, ag_masks, _, err = PdbParser.load(
        args.antigen, chain_id=chain_id_ag, aa_seq=ag_seq
    )
    if err:
        raise RuntimeError(f"Failed to load antigen: {err}")
    if len(ag_seq_pdb) != len(ag_seq):
        raise ValueError(
            f"Antigen length mismatch: PDB has {len(ag_seq_pdb)}, FASTA has {len(ag_seq)}"
        )

    # Build full complex sequence and coordinates
    full_seq = ab_seq + ag_seq
    full_coords = torch.cat([fv_coords, ag_coords], dim=0)  # [L, 14, 3]
    full_masks = torch.cat([fv_masks, ag_masks], dim=0)     # [L, 14]

    # Infer CDR spans from X positions in FASTA
    design_mask_h = torch.tensor([1 if c == "X" else 0 for c in h_seq], dtype=torch.int8)
    design_mask_l = torch.tensor([1 if c == "X" else 0 for c in l_seq], dtype=torch.int8) if l_seq else torch.zeros(0, dtype=torch.int8)
    design_mask_ab = torch.cat([design_mask_h, design_mask_l], dim=0)
    design_mask_ag = torch.zeros(len(ag_seq), dtype=torch.int8)
    mask_design = torch.cat([design_mask_ab, design_mask_ag], dim=0)

    cdr_sequences = _infer_cdr_sequences_from_mask(
        design_mask_h, design_mask_l, args.design_loops
    )

    # Build region metadata
    seq_lengths = {"H": len(h_seq), "A": len(ag_seq)}
    if l_seq:
        seq_lengths["L"] = len(l_seq)

    region_metadata = build_antibody_region_metadata(
        sequence_lengths=seq_lengths,
        cdr_sequences=cdr_sequences,
        atom_mask=full_masks,
    )

    # Build atom14 supervision with placeholder sequence (CDR positions will be overwritten)
    # Replace X with A as placeholder (gets overwritten anyway)
    seq_for_sup = full_seq.replace("X", "A")
    atom14_sync = Atom14SeqSync()
    atom14_sup = atom14_sync.build_supervision(
        seq=seq_for_sup,
        cord_n14_tf=full_coords,
        cmsk_n14_tf=full_masks,
        cdr_mask=region_metadata["cdr_mask"],
    )

    # CRITICAL: Zero out CDR coordinates and set masks to all-14 (leak-free, see ADR 0003)
    cdr_mask_bool = region_metadata["cdr_mask"].to(torch.bool)
    atom14_sup["cords_atom14"][cdr_mask_bool] = 0.0
    atom14_sup["cmsk_atom14"][cdr_mask_bool] = 1.0  # all 14 atoms "valid" (marker convention)

    # Epitope
    epitope = _build_epitope(args, ag_seq, ag_coords, fv_coords, len(ab_seq))

    # asym_id
    asym_id = get_asym_ids(seq_lengths)

    # Antibody/antigen masks
    ab_len = len(ab_seq)
    mask_ab = torch.zeros(len(full_seq), dtype=torch.bool)
    mask_ab[:ab_len] = True

    # Center on antigen centroid (match training convention)
    ag_ca = ProtStruct.get_atoms(ag_seq, ag_coords, ["CA"])
    ag_ca_valid = ag_ca[ag_masks[len(ab_seq):, 1].to(torch.bool)]
    if len(ag_ca_valid) == 0:
        raise RuntimeError("No valid CA atoms in antigen")
    ag_centroid = ag_ca_valid.mean(dim=0)
    atom14_sup["cords_atom14"] -= ag_centroid

    # Pack into prot_data_curr format
    prot_data_curr = {
        "seq": full_seq,
        "cord": atom14_sup["cords_atom14"][:, 1, :],  # CA coords [L, 3]
        "cords_atom14": atom14_sup["cords_atom14"],
        "cmsk": atom14_sup["cmsk_atom14"][:, 1],      # CA mask [L]
        "cmsk_atom14": atom14_sup["cmsk_atom14"],
        "mask_design": mask_design,
        "mask_ab": mask_ab,
        "asym_id": asym_id,
        "a-cord": ag_coords - ag_centroid,
        "a-cmsk": ag_masks,
        "epitope": epitope,
        "contact": None,  # Always None for inference (3-dim epitope-only)
        **_region_metadata_for_model(region_metadata),
    }

    # Move to device and add batch dim
    for k, v in prot_data_curr.items():
        if torch.is_tensor(v):
            prot_data_curr[k] = v.unsqueeze(0).to(device)
        elif k == "seq":
            pass  # keep as string

    logger.info(
        f"Built prot_data_curr: {len(h_seq)}H + {len(l_seq) if l_seq else 0}L + "
        f"{len(ag_seq)}A, {int(mask_design.sum())} design positions"
    )
    logger.info(f"Inferred CDR spans: {cdr_sequences}")

    return prot_data_curr


def _infer_cdr_sequences_from_mask(
    design_mask_h: torch.Tensor,
    design_mask_l: torch.Tensor,
    explicit_loops: Optional[str],
) -> Dict[str, List[int]]:
    """Map X spans to CDR loop names.

    WARNING: The default positional mapping assumes all 6 CDRs are designed in
    order (H1, H2, H3, L1, L2, L3).  If only H3 is designed, the first span
    would be incorrectly labeled H1.  Use --design_loops to override.
    """
    loop_names = []
    if len(design_mask_h) > 0:
        loop_names.extend(["H1", "H2", "H3"])
    if len(design_mask_l) > 0:
        loop_names.extend(["L1", "L2", "L3"])

    # Extract contiguous spans
    spans_h = _extract_spans(design_mask_h)
    spans_l = _extract_spans(design_mask_l, offset=len(design_mask_h))
    all_spans = spans_h + spans_l

    if explicit_loops:
        loop_names_explicit = [x.strip() for x in explicit_loops.split(",")]
        if len(loop_names_explicit) != len(all_spans):
            raise ValueError(
                f"--design_loops has {len(loop_names_explicit)} names but "
                f"FASTA has {len(all_spans)} X spans"
            )
        loop_names = loop_names_explicit

    if len(all_spans) > len(loop_names):
        raise ValueError(
            f"Found {len(all_spans)} X spans but only {len(loop_names)} loop names. "
            f"Use --design_loops to specify explicitly."
        )

    cdr_sequences = {}
    for loop_name, span in zip(loop_names, all_spans):
        # Convert to 1-based indices for the respective chain
        if loop_name.startswith("H"):
            indices = [i + 1 for i in range(span[0], span[1])]
        else:  # L
            indices = [i - len(design_mask_h) + 1 for i in range(span[0], span[1])]
        cdr_sequences[f"cdr_{loop_name}"] = indices

    return cdr_sequences


def _extract_spans(mask: torch.Tensor, offset: int = 0) -> List[Tuple[int, int]]:
    """Find contiguous 1-spans in mask, return as [(start, end), ...] with offset."""
    if len(mask) == 0:
        return []
    spans = []
    in_span = False
    start = 0
    for i, val in enumerate(mask.tolist()):
        if val and not in_span:
            start = i + offset
            in_span = True
        elif not val and in_span:
            spans.append((start, i + offset))
            in_span = False
    if in_span:
        spans.append((start, len(mask) + offset))
    return spans


def _build_epitope(
    args, ag_seq: str, ag_coords: torch.Tensor, fv_coords: torch.Tensor, ab_len: int
) -> torch.Tensor:
    """Build epitope vector (1 = epitope residue, 0 = not)."""
    if args.epitope:
        indices = [int(x.strip()) for x in args.epitope.split(",")]
        epitope = torch.zeros(len(ag_seq), dtype=torch.int8)
        for idx in indices:
            if idx < 1 or idx > len(ag_seq):
                raise ValueError(f"Epitope index {idx} out of range [1, {len(ag_seq)}]")
            epitope[idx - 1] = 1
        logger.info(f"Epitope: {len(indices)} residues from --epitope")
    else:
        # cal_epitope: antigen residues within DIST of any Fv atom
        dist_thres = args.cal_epitope
        ag_ca = ProtStruct.get_atoms(ag_seq, ag_coords, ["CA"])
        fv_ca = fv_coords[:, 1, :]  # CA
        from IgGM.utils import cdist
        dist_mat = cdist(ag_ca.unsqueeze(0), fv_ca.unsqueeze(0))[0]
        epitope_bool = (dist_mat.min(dim=1)[0] < dist_thres)
        epitope = epitope_bool.to(torch.int8)
        logger.info(f"Epitope: {int(epitope.sum())} residues within {dist_thres} Å of Fv")

    return epitope


def _region_metadata_for_model(region_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten region metadata (same as data_module.py:216)."""
    return {
        "loop_global_res_indices": region_metadata["loop_global_res_indices"],
        "loop_local_res_indices": region_metadata["loop_local_res_indices"],
        "loop_anchor_local_indices": region_metadata["loop_anchor_local_indices"],
        "loop_true_lens": region_metadata["loop_true_lens"],
    }


def export_results(
    results: List[Dict[str, Any]],
    prot_data_curr: Dict[str, Any],
    args,
    atom14_sync: Atom14SeqSync,
):
    """Export PDB and FASTA for each sample."""
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    full_seq = prot_data_curr["seq"]
    cdr_mask = prot_data_curr["cdr_mask"][0]  # remove batch dim

    for i, result in enumerate(results):
        # Decode CDR sequence from geometry (this is the reported sequence)
        cord_pred = result["cord"].cpu()
        cmsk_pred = torch.ones_like(cord_pred[:, :, 0])  # assume all atoms present

        seq_geo = atom14_sync.decode_cdr_sequence(
            seq_true=full_seq,
            pred_cord_n14_tf=cord_pred,
            pred_cmsk_n14_tf=cmsk_pred,
            cdr_mask=cdr_mask,
        )

        seq_head = result.get("seq_head", full_seq)

        # Export PDB
        pdb_path = f"{args.output}_sample{i}.pdb"
        # Build cmsk from decoded sequence
        cmsk_geo = ProtStruct.get_cmsk_vld(seq_geo, cord_pred.device)

        # Split by chain
        seq_lengths = {
            "H": int((prot_data_curr["asym_id"][0] == 2).sum()),
            "A": int((prot_data_curr["asym_id"][0] == 0).sum()),
        }
        if (prot_data_curr["asym_id"][0] == 1).any():
            seq_lengths["L"] = int((prot_data_curr["asym_id"][0] == 1).sum())

        h_len = seq_lengths["H"]
        l_len = seq_lengths.get("L", 0)
        ab_len = h_len + l_len

        prot_data_export = {
            "H": {
                "seq": seq_geo[:h_len],
                "cord": cord_pred[:h_len],
                "cmsk": cmsk_geo[:h_len],
            },
            "A": {
                "seq": seq_geo[ab_len:],
                "cord": cord_pred[ab_len:],
                "cmsk": cmsk_geo[ab_len:],
            },
        }
        if l_len > 0:
            prot_data_export["L"] = {
                "seq": seq_geo[h_len:ab_len],
                "cord": cord_pred[h_len:ab_len],
                "cmsk": cmsk_geo[h_len:ab_len],
            }

        PdbParser.save_multimer(prot_data_export, pdb_path, pred_info=f"AbG_sample{i}")
        logger.info(f"Saved {pdb_path}")

        # Export FASTA (both sequences for comparison)
        fasta_path = f"{args.output}_sample{i}.fasta"
        with open(fasta_path, "w") as f:
            f.write(f">geo_decoded | CDR types from atom14 geometry (reported)\n")
            f.write(f"{seq_geo}\n")
            f.write(f">seq_head | sequence-head feedback chain (auxiliary)\n")
            f.write(f"{seq_head}\n")
        logger.info(f"Saved {fasta_path}")


def compute_metrics(
    results: List[Dict[str, Any]],
    prot_data_curr: Dict[str, Any],
    args,
    atom14_sync: Atom14SeqSync,
) -> Dict[str, Any]:
    """Compute structural metrics against reference PDB."""
    if not args.ref_pdb:
        return {}

    logger.info(f"Computing metrics against {args.ref_pdb}")

    # Load reference
    full_seq = prot_data_curr["seq"]
    ref_seq, ref_cord, ref_cmsk, _, err = PdbParser.load(args.ref_pdb, aa_seq=full_seq)
    if err:
        raise RuntimeError(f"Failed to load reference PDB: {err}")

    metrics_calculator = StructureMetrics(MetricConfig())

    # Extract region info
    seq_lengths = {
        "H": int((prot_data_curr["asym_id"][0] == 2).sum()),
        "A": int((prot_data_curr["asym_id"][0] == 0).sum()),
    }
    if (prot_data_curr["asym_id"][0] == 1).any():
        seq_lengths["L"] = int((prot_data_curr["asym_id"][0] == 1).sum())

    # Decode CDR sequences and compute metrics for each sample
    all_metrics = []
    for i, result in enumerate(results):
        cord_pred = result["cord"].cpu()
        cmsk_pred = torch.ones_like(cord_pred[:, :, 0])

        seq_pred = atom14_sync.decode_cdr_sequence(
            seq_true=full_seq,
            pred_cord_n14_tf=cord_pred,
            pred_cmsk_n14_tf=cmsk_pred,
            cdr_mask=prot_data_curr["cdr_mask"][0],
        )

        sample_metrics = metrics_calculator(
            pred_seq=seq_pred,
            pred_cord=cord_pred,
            true_seq=ref_seq,
            tgt_cord=ref_cord,
            asym_id=prot_data_curr["asym_id"],
            seq_lengths=seq_lengths,
            cdr_sequences={},  # Will be inferred from X spans
            native_atom_mask=ref_cmsk,
            antibody_mask=prot_data_curr["mask_ab"],
            antigen_mask=prot_data_curr["antigen_mask"],
            cdr_mask=prot_data_curr["cdr_mask"],
        )
        all_metrics.append(sample_metrics)

    # Aggregate: mean across samples
    aggregated = {}
    if all_metrics:
        for key in all_metrics[0]:
            if torch.is_tensor(all_metrics[0][key]):
                aggregated[key] = torch.stack([m[key] for m in all_metrics]).mean().item()

    # Add top-1 (best TM-score)
    if "tm_score" in aggregated:
        tm_scores = [m["tm_score"].item() for m in all_metrics]
        aggregated["tm_score_top1"] = max(tm_scores)

    logger.info(f"Metrics (mean over {len(results)} samples): {aggregated}")
    return aggregated


def main():
    args = parse_args()

    # Load models
    design_model, plm_featurizer, diffuser = load_models(args)

    # Build input
    prot_data_curr = build_prot_data_curr(args)

    # Run reverse sampling
    results = []
    for i in range(args.num_samples):
        seed = args.seed + i
        logger.info(f"Sample {i+1}/{args.num_samples}, seed={seed}")

        result = run_reverse_sampling(
            model=design_model,
            plm_featurizer=plm_featurizer,
            diffuser=diffuser,
            prot_data_curr=prot_data_curr,
            n_sample_steps=args.steps,
            seed=seed,
            seq_feedback=args.seq_feedback,
        )
        results.append(result)
        logger.info(f"  Final step: t={result['final_step']}, schedule: {result['schedule']}")

    # Export
    atom14_sync = Atom14SeqSync()
    export_results(results, prot_data_curr, args, atom14_sync)

    # Metrics
    if args.ref_pdb:
        metrics = compute_metrics(results, prot_data_curr, args, atom14_sync)
        metrics_path = f"{args.output}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved {metrics_path}")

    logger.info("Inference complete")


if __name__ == "__main__":
    main()
