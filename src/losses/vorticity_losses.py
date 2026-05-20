"""Vorticity losses."""

from __future__ import annotations

import torch

from src.physics.vorticity import vorticity_transport_residual


def vorticity_mse(omega_pred: torch.Tensor, omega_ref: torch.Tensor) -> torch.Tensor:
    return torch.mean((omega_pred - omega_ref).pow(2))


def vorticity_transport_loss(model: torch.nn.Module, coords: torch.Tensor, nu: float, steady: bool = False) -> torch.Tensor:
    r = vorticity_transport_residual(model, coords, nu=nu, steady=steady)
    return torch.mean(r.pow(2))

