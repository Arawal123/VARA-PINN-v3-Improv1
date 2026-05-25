"""Streamfunction-pressure model for incompressible cavity PINNs."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .mlp import MLP


class StreamfunctionMLP(nn.Module):
    """Network that predicts psi and p, then exposes velocity-pressure outputs.

    The public forward signature intentionally returns ``[u, v, p]`` so the
    existing Navier-Stokes residual, loss, metric, and controller code can be
    reused without special casing.
    """

    requires_input_grad_forward = True

    def __init__(
        self,
        in_dim: int = 2,
        hidden_layers: Iterable[int] = (96, 96, 96, 96),
        activation: str = "tanh",
        input_lower: Iterable[float] | None = None,
        input_upper: Iterable[float] | None = None,
        lid_velocity: float = 1.0,
        boundary_transform: bool = True,
        enforce_lid_profile: bool = False,
        corner_eps: float = 0.08,
        include_vorticity_transport_loss: bool = False,
    ) -> None:
        super().__init__()
        if in_dim != 2:
            raise ValueError("StreamfunctionMLP currently supports steady 2D coordinates only.")
        self.net = MLP(
            in_dim=in_dim,
            out_dim=2,
            hidden_layers=hidden_layers,
            activation=activation,
            input_lower=input_lower,
            input_upper=input_upper,
        )
        lower = list(input_lower or [0.0, 0.0])
        upper = list(input_upper or [1.0, 1.0])
        self.register_buffer("input_lower", torch.tensor(lower, dtype=torch.float32))
        self.register_buffer("input_upper", torch.tensor(upper, dtype=torch.float32))
        self.lid_velocity = float(lid_velocity)
        self.boundary_transform = bool(boundary_transform)
        self.enforce_lid_profile = bool(enforce_lid_profile)
        self.corner_eps = float(corner_eps)
        self.include_vorticity_transport_loss = bool(include_vorticity_transport_loss)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            if coords.requires_grad:
                x = coords
            else:
                x = coords.detach().clone().requires_grad_(True)
            psi, p = self.psi_pressure(x)
            grad_psi = torch.autograd.grad(
                psi,
                x,
                grad_outputs=torch.ones_like(psi),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            u = grad_psi[:, 1:2]
            v = -grad_psi[:, 0:1]
            return torch.cat([u, v, p], dim=1)

    def psi_pressure(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(coords)
        psi_raw = raw[:, 0:1]
        p = raw[:, 1:2]
        if not self.boundary_transform:
            return psi_raw, p
        return self._cavity_streamfunction(coords, psi_raw), p

    def _cavity_streamfunction(self, coords: torch.Tensor, psi_raw: torch.Tensor) -> torch.Tensor:
        x0, y0 = self.input_lower[0], self.input_lower[1]
        x1, y1 = self.input_upper[0], self.input_upper[1]
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        xi = ((x - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0.0, 1.0)
        eta = ((y - y0) / (y1 - y0).clamp_min(1e-12)).clamp(0.0, 1.0)
        wall_distance = xi * (1.0 - xi) * eta * (1.0 - eta)
        if not self.enforce_lid_profile:
            return wall_distance * psi_raw

        correction = wall_distance.pow(2) * psi_raw
        lid_shape = self._smooth_lid_shape(xi)
        lid_profile = eta.pow(2) * (1.0 - eta)
        lid_streamfunction = -self.lid_velocity * (y1 - y0) * lid_shape * lid_profile
        return lid_streamfunction + correction

    def _smooth_lid_shape(self, xi: torch.Tensor) -> torch.Tensor:
        eps = max(float(self.corner_eps), 1e-6)
        left = _smoothstep((xi / eps).clamp(0.0, 1.0))
        right = _smoothstep(((1.0 - xi) / eps).clamp(0.0, 1.0))
        return left * right


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    return t * t * (3.0 - 2.0 * t)
