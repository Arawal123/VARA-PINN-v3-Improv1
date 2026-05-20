"""Pressure Poisson auxiliary residuals."""

from __future__ import annotations

import torch

from .navier_stokes import gradients, navier_stokes_residuals


def pressure_poisson_residual(model: torch.nn.Module, coords: torch.Tensor, nu: float) -> torch.Tensor:
    """Approximate incompressible pressure Poisson residual."""
    res = navier_stokes_residuals(model, coords, nu=nu, steady=True)
    p = res["p"]
    x = res["coords"]
    grad_p = gradients(p, x)
    p_xx = gradients(grad_p[:, 0:1], x)[:, 0:1]
    p_yy = gradients(grad_p[:, 1:2], x)[:, 1:2]
    rhs = -(
        res["u_x"] * res["u_x"]
        + 2.0 * res["u_y"] * res["v_x"]
        + res["v_y"] * res["v_y"]
    )
    return p_xx + p_yy - rhs

