# -*- coding: utf-8 -*-
"""Factory helpers for building train/infer modules as plain ``torch.nn.Module``.

These helpers centralize instantiation paths that were previously embedded in
``IgGM/model/arch/design_model/model.py`` and ``IgGM/deploy/ab_design.py``.
The returned modules are unchanged model classes, preserving parameter names and
``state_dict`` compatibility with released checkpoints.
"""

from __future__ import annotations

from typing import Optional, Tuple

from torch import nn

from .arch import DesignModel, PPIModel


def build_ppi_featurizer_module(ppi_path: str) -> nn.Module:
    """Build the sequence featurizer as a pure ``nn.Module``."""

    return PPIModel.restore(ppi_path)


def build_design_model_module(design_path: str, config) -> nn.Module:
    """Build the design trunk as a pure ``nn.Module``.

    Notes:
        - This function intentionally delegates to ``DesignModel.restore`` so
          checkpoint ``state_dict`` keys stay identical to legacy loading.
    """

    return DesignModel.restore(design_path, config)


def build_iggm_modules(
    *,
    ppi_path: str,
    design_path: str,
    config,
) -> Tuple[nn.Module, nn.Module, Optional[int], Optional[int]]:
    """Build IgGM featurizer + design modules and return dims for config sync."""

    plm_featurizer = build_ppi_featurizer_module(ppi_path)
    c_s = getattr(plm_featurizer, "c_s", None)
    c_p = getattr(plm_featurizer, "c_z", None)
    if c_s is not None:
        config.c_s = c_s
    if c_p is not None:
        config.c_p = c_p
    design_model = build_design_model_module(design_path, config)
    return plm_featurizer, design_model, c_s, c_p
