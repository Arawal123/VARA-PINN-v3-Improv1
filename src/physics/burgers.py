"""Placeholder Burgers utilities for future scalar PDE benchmarks."""

from __future__ import annotations

import torch


def burgers_residual(_model: torch.nn.Module, _coords: torch.Tensor, _nu: float) -> torch.Tensor:
    """Burgers support is intentionally deferred behind this explicit stub."""
    raise NotImplementedError("Burgers benchmark is not part of the first Kovasznay target.")

