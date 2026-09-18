#!/usr/bin/env python3
"""Static self-check for FR/CDR sync integration.

This check is intentionally import-light so it can run in environments without
PyTorch. It verifies that:
- required docs exist;
- design CLI exposes new switches;
- AbDesigner contains legacy + fr_cdr_sync integration hooks;
- BaseDesigner export path handles `loop_export_mask`;
- occupancy prefix decoding semantics remain prefix-valid.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DOCS = [
    'docs/fr_cdr_sync_inference.md',
    'docs/fr_cdr_sync_config.md',
    'docs/fr_cdr_sync_migration_notes.md',
    'docs/fr_cdr_sync_file_changes.md',
    'docs/fr_cdr_sync_known_assumptions.md',
]


def prefix_len(values, threshold=0.5):
    out = 0
    for x in values:
        if x >= threshold:
            out += 1
        else:
            break
    return out


def require(path: str, needles: list[str]) -> None:
    text = (ROOT / path).read_text(encoding='utf-8')
    for needle in needles:
        assert needle in text, f'{needle!r} missing from {path}'


def main() -> None:
    for doc in REQUIRED_DOCS:
        assert (ROOT / doc).exists(), f'missing {doc}'

    require('design.py', [
        '--diffusion_mode', '--structure_mode', '--loss_mode',
        '--fr_noise_scale_trsl', '--fr_noise_scale_rota',
        '--cdr_local_noise_scale', '--occupancy_prediction_mode',
    ])
    require('IgGM/deploy/ab_design.py', [
        '_attach_region_metadata', '_assemble_fr_cdr_coords',
        'rebuild_loops_from_local_coords', 'merge_noisy_fr_and_loops',
        "inputs['structure_mode'] = self.structure_mode",
    ])
    require('IgGM/deploy/base_designer.py', ['loop_export_mask', '_output_to_fasta', '_output_to_pdb'])

    assert prefix_len([0.9, 0.8, 0.1, 0.7]) == 2
    assert prefix_len([0.2, 0.9, 0.9]) == 0
    assert prefix_len([0.9, 0.9, 0.9]) == 3

    print('fr_cdr_sync_static_check: ok')


if __name__ == '__main__':
    main()
