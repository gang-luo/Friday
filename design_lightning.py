#!/usr/bin/env python3
"""Lightweight inference entrypoint for IgGM checkpoints (no training logic)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from IgGM.deploy import AbDesigner
from IgGM.model.pretrain import IGSO3Buffer_trunk, esm_ppi_650m_ab
from IgGM.protein import cal_ppi, crop_sequence_with_epitope
from IgGM.protein.parser import PdbParser, parse_fasta
from IgGM.utils import setup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure inference script for IgGM")
    parser.add_argument("--fasta", required=True, help="Input FASTA with H/L/A chain sequences")
    parser.add_argument("--antigen", required=True, help="Antigen PDB path")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--design_ckpt", required=True, help="Design model checkpoint path")
    parser.add_argument("--ppi_ckpt", default="", help="Optional PPI checkpoint path")
    parser.add_argument("--igso3_buffer", default="", help="Optional IGSO3 buffer path")
    parser.add_argument("--steps", type=int, default=10, help="Diffusion sampling steps")
    parser.add_argument("--chunk_size", type=int, default=64, help="Inference chunk size")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_antigen_size", type=int, default=2000)
    parser.add_argument("--run_task", choices=["design", "inverse_design", "fr_design", "affinity_maturation"], default="design")
    parser.add_argument("--epitope", nargs="+", type=int, default=None)
    parser.add_argument("--cal_epitope", action="store_true", default=False)
    parser.add_argument("--relax", action="store_true")
    parser.add_argument("--diffusion_mode", type=str, default="fr_cdr_sync", choices=["legacy", "fr_cdr_sync"])
    parser.add_argument("--structure_mode", type=str, default="fr_cdr_sync", choices=["legacy", "fr_cdr_sync"])
    parser.add_argument("--occupancy_prediction_mode", type=str, default="joint_predict")
    parser.add_argument("--occupancy_threshold", type=float, default=0.5)
    return parser.parse_args()


def build_chains(args: argparse.Namespace):
    sequences, ids, _ = parse_fasta(args.fasta)
    assert len(sequences) in (1, 2, 3), "fasta must contain 1/2/3 chains"
    if args.cal_epitope:
        epitope = cal_ppi(args.antigen, ids, sequences)
        print("epitope:", " ".join(str(int(i) + 1) for i in torch.nonzero(epitope).flatten()))
        return None, None

    chains = [{"sequence": seq, "id": seq_id} for seq, seq_id in zip(sequences, ids) if seq_id != ids[-1]]
    aa_seq, atom_cord, atom_cmsk, _, _ = PdbParser.load(args.antigen, chain_id=ids[-1], aa_seq=sequences[-1])

    if args.epitope is None:
        epitope = cal_ppi(args.antigen, ids, sequences)
    else:
        epitope = torch.zeros(len(aa_seq))
        for pos in args.epitope:
            epitope[pos - 1] = 1

    if len(aa_seq) > args.max_antigen_size:
        aa_seq, atom_cord, atom_cmsk, epitope, _ = crop_sequence_with_epitope(
            aa_seq, atom_cord, atom_cmsk, epitope, max_len=args.max_antigen_size
        )

    chains.append({"sequence": aa_seq, "cord": atom_cord, "cmsk": atom_cmsk, "epitope": epitope, "id": ids[-1]})
    name = Path(args.fasta).stem
    return chains, name


def main() -> None:
    args = parse_args()
    setup(True)

    chains, name = build_chains(args)
    if chains is None:
        return

    Path(args.output).mkdir(parents=True, exist_ok=True)
    designer = AbDesigner(
        ppi_path=(args.ppi_ckpt or esm_ppi_650m_ab()),
        design_path=args.design_ckpt,
        buffer_path=(args.igso3_buffer or IGSO3Buffer_trunk()),
        config=args,
    )
    designer.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    for idx in range(args.num_samples):
        out_pdb = Path(args.output) / f"{name}_{idx}.pdb"
        designer.infer_pdb(
            chains,
            filename=str(out_pdb),
            relax=args.relax,
            task=args.run_task,
            chunk_size=args.chunk_size,
            temperature=args.temperature,
        )
        print(f"saved: {out_pdb}")


if __name__ == "__main__":
    main()