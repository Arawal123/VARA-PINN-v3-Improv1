"""Vorticity utilities."""

from __future__ import annotations

import torch

from .navier_stokes import navier_stokes_residuals


def compute_vorticity(model: torch.nn.Module, coords: torch.Tensor, nu: float, steady: bool = True) -> torch.Tensor:
    """Return omega = dv/dx - du/dy."""
    return navier_stokes_residuals(model, coords, nu=nu, steady=steady)["omega"]


def vorticity_transport_residual(
    model: torch.nn.Module,
    coords: torch.Tensor,
    nu: float,
    steady: bool = False,
) -> torch.Tensor:
    """Compute a 2D vorticity transport residual."""
    res = navier_stokes_residuals(model, coords, nu=nu, steady=steady)
    omega = res["omega"]
    xyt = res["coords"]
    grad_omega = torch.autograd.grad(
        omega,
        xyt,
        grad_outputs=torch.ones_like(omega),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    omega_x, omega_y = grad_omega[:, 0:1], grad_omega[:, 1:2]
    omega_t = torch.zeros_like(omega) if steady or xyt.shape[1] < 3 else grad_omega[:, 2:3]
    omega_xx = torch.autograd.grad(
        omega_x,
        xyt,
        grad_outputs=torch.ones_like(omega_x),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0][:, 0:1]
    omega_yy = torch.autograd.grad(
        omega_y,
        xyt,
        grad_outputs=torch.ones_like(omega_y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0][:, 1:2]
    return omega_t + res["u"] * omega_x + res["v"] * omega_y - nu * (omega_xx + omega_yy)

