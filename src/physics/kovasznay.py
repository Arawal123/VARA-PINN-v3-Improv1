"""Kovasznay flow benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np
import torch


@dataclass(frozen=True)
class KovasznayFlow:
    """Analytical steady 2D incompressible Navier-Stokes solution."""

    reynolds: float = 40.0
    x_min: float = -0.5
    x_max: float = 1.0
    y_min: float = -0.5
    y_max: float = 1.5

    @property
    def nu(self) -> float:
        return 1.0 / self.reynolds

    @property
    def lambda_value(self) -> float:
        re = self.reynolds
        return re / 2.0 - sqrt((re * re) / 4.0 + 4.0 * pi * pi)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    def exact_np(self, xy: np.ndarray) -> dict[str, np.ndarray]:
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        lam = self.lambda_value
        u = 1.0 - np.exp(lam * x) * np.cos(2.0 * pi * y)
        v = (lam / (2.0 * pi)) * np.exp(lam * x) * np.sin(2.0 * pi * y)
        p = 0.5 * (1.0 - np.exp(2.0 * lam * x))
        omega = ((lam * lam) / (2.0 * pi) - 2.0 * pi) * np.exp(lam * x) * np.sin(2.0 * pi * y)
        px = -lam * np.exp(2.0 * lam * x)
        py = np.zeros_like(px)
        speed = np.sqrt(u * u + v * v)
        return {"u": u, "v": v, "p": p, "omega": omega, "p_x": px, "p_y": py, "speed": speed}

    def exact_torch(self, xy: torch.Tensor) -> dict[str, torch.Tensor]:
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        lam = torch.as_tensor(self.lambda_value, dtype=xy.dtype, device=xy.device)
        pi_t = torch.as_tensor(pi, dtype=xy.dtype, device=xy.device)
        u = 1.0 - torch.exp(lam * x) * torch.cos(2.0 * pi_t * y)
        v = (lam / (2.0 * pi_t)) * torch.exp(lam * x) * torch.sin(2.0 * pi_t * y)
        p = 0.5 * (1.0 - torch.exp(2.0 * lam * x))
        omega = ((lam * lam) / (2.0 * pi_t) - 2.0 * pi_t) * torch.exp(lam * x) * torch.sin(2.0 * pi_t * y)
        px = -lam * torch.exp(2.0 * lam * x)
        py = torch.zeros_like(px)
        speed = torch.sqrt(u * u + v * v)
        return {"u": u, "v": v, "p": p, "omega": omega, "p_x": px, "p_y": py, "speed": speed}

    def grid(self, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(self.x_min, self.x_max, nx)
        y = np.linspace(self.y_min, self.y_max, ny)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
        return X, Y, xy


def center_pressure(p: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Remove the mean pressure gauge."""
    return p - p.mean()

