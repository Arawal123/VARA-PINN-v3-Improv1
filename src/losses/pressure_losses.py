"""Pressure-specific losses."""

from __future__ import annotations

import torch

from src.physics.kovasznay import center_pressure
from src.physics.pressure_poisson import pressure_poisson_residual


def pressure_anchor_loss(p_pred: torch.Tensor, anchor_value: float = 0.0) -> torch.Tensor:
    """Gauge anchor on mean pressure."""
    return torch.mean((torch.mean(p_pred) - float(anchor_value)) ** 2)


def pressure_centered_mse(p_pred: torch.Tensor, p_ref: torch.Tensor) -> torch.Tensor:
    """Mean-centered pressure loss."""
    return torch.mean((center_pressure(p_pred) - center_pressure(p_ref)).pow(2))


def pressure_poisson_loss(model: torch.nn.Module, coords: torch.Tensor, nu: float) -> torch.Tensor:
    """Mean-square pressure Poisson residual."""
    r = pressure_poisson_residual(model, coords, nu=nu)
    return torch.mean(r.pow(2))

