"""Convert SAbDab entries to an IgGM-compatible sample dictionary."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

from IgGM.protein.data_transform.processing_multimer import calc_ppi_sites, get_asym_ids

from .structure import load_chain_structure


def _chain_dict(chain_id: str, seq: str, cord: torch.Tensor, cmsk: torch.Tensor) -> Dict[str, object]:
    return {"id": chain_id, "seq": seq, "sequence": seq, "cord": cord, "cmsk": cmsk}


def _calc_contact_and_epitope(
    heavy_chain: Dict[str, object],
    antigen_chain: Dict[str, object],
    light_chain: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    prot_data = {heavy_chain["id"]: heavy_chain, antigen_chain["id"]: antigen_chain}
    if light_chain is not None:
        prot_data[light_chain["id"]] = light_chain
        ab_seq = heavy_chain["sequence"] + light_chain["sequence"]
        contact_h = calc_ppi_sites(prot_data, [heavy_chain["id"], antigen_chain["id"]])[heavy_chain["id"]]
        contact_l = calc_ppi_sites(prot_data, [light_chain["id"], antigen_chain["id"]])[light_chain["id"]]
        contact = torch.cat([contact_h, contact_l], dim=0)
    else:
        ab_seq = heavy_chain["sequence"]
        contact = calc_ppi_sites(prot_data, [heavy_chain["id"], antigen_chain["id"]])[heavy_chain["id"]]

    epitope = torch.zeros(len(antigen_chain["sequence"]), dtype=torch.int8)
    if torch.any(contact > 0):
        ppi_sites = calc_ppi_sites(prot_data, [heavy_chain["id"], antigen_chain["id"]])[antigen_chain["id"]]
        if light_chain is not None:
            ppi_sites_l = calc_ppi_sites(prot_data, [light_chain["id"], antigen_chain["id"]])[antigen_chain["id"]]
            ppi_sites = torch.maximum(ppi_sites, ppi_sites_l)
        epitope = ppi_sites.to(torch.int8)

    mask_ab = torch.ones(len(ab_seq), dtype=torch.int8)
    return contact.to(torch.int8), epitope, mask_ab


def convert_entry_to_sample(
    entry: Dict[str, object],
    pdb_path: str | Path,
    *,
    require_light_chain: bool = False,
    max_antigen_length: Optional[int] = None,
    compute_contact: bool = True,
) -> Optional[Dict[str, object]]:
    """Map one SAbDab metadata entry to a unified sample dict.

    Supports both antibody (H+L+Ag) and nanobody (H+Ag) records.
    Returns None when entry is filtered by configured rules.
    """

    heavy_id = entry.get("heavy_chain_id")
    light_id = entry.get("light_chain_id")
    antigen_ids: Iterable[str] = entry.get("antigen_chain_ids", [])

    if not heavy_id or not antigen_ids:
        return None
    if require_light_chain and not light_id:
        return None

    antigen_id = next(iter(antigen_ids))
    heavy = load_chain_structure(pdb_path, str(heavy_id))
    light = load_chain_structure(pdb_path, str(light_id)) if light_id else None
    antigen = load_chain_structure(pdb_path, str(antigen_id))
    # sequence info extrct, but may has error for cords.shape not match seq-length, we using try-except to drop the error samples durining training

    if max_antigen_length is not None and len(antigen["seq"]) > max_antigen_length:
        return None

    chains = [_chain_dict(str(heavy_id), heavy["seq"], heavy["cord"], heavy["cmsk"])]
    if light is not None:
        chains.append(_chain_dict(str(light_id), light["seq"], light["cord"], light["cmsk"]))
    chains.append(_chain_dict(str(antigen_id), antigen["seq"], antigen["cord"], antigen["cmsk"]))

    sequences = [chain["sequence"] for chain in chains]
    concatenated_seq = "".join(sequences)
    concatenated_cord = torch.cat([chain["cord"] for chain in chains], dim=0)
    concatenated_cmsk = torch.cat([chain["cmsk"] for chain in chains], dim=0)

    asym_id = len(chains) - get_asym_ids(sequences)
    mask_ab = torch.zeros(len(concatenated_seq), dtype=torch.int8)
    mask_ab[:-len(chains[-1]["sequence"])] = 1
    mask_design = torch.zeros(len(concatenated_seq), dtype=torch.int8)

    if compute_contact:
        contact, epitope, _ = _calc_contact_and_epitope(
            chains[0], chains[-1], chains[1] if len(chains) == 3 else None
        )
    else:
        ab_len = len(chains[0]["sequence"]) + (len(chains[1]["sequence"]) if len(chains) == 3 else 0)
        contact = torch.zeros(ab_len, dtype=torch.int8)
        epitope = torch.zeros(len(chains[-1]["sequence"]), dtype=torch.int8)

    chains[-1]["epitope"] = epitope
    chains[-1]["contact"] = contact

    sample_id = f"{entry['pdb_id']}_{'_'.join([x['id'] for x in chains])}"
    return {
        "sample_id": sample_id,
        "schema_version": 1,
        "entry": entry,
        "base": {
            "chain_ids": [chain["id"] for chain in chains],
            "complex_id": ":".join(chain["id"] for chain in chains),
        },
        "chains": chains,
        "complex": {
            "seq": concatenated_seq,
            "cord": concatenated_cord,
            "cmsk": concatenated_cmsk,
            "asym_id": asym_id.unsqueeze(0),
            "mask_ab": mask_ab,
            "mask_design": mask_design,
            "a-cord": chains[-1]["cord"],
            "a-cmsk": chains[-1]["cmsk"],
            "epitope": epitope,
            "contact": contact,
        },
        "metadata": {
            "filters": {
                "require_light_chain": require_light_chain,
                "max_antigen_length": max_antigen_length,
                "compute_contact": compute_contact,
            },
            "is_nanobody": light is None,
        },
    }
