"""Rectangular benchmark definitions for professor-facing PINN comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
import torch


@dataclass(frozen=True)
class RectangularBenchmarkBase:
    """Common rectangular benchmark interface used by the existing trainer."""

    reynolds: float = 40.0
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    amplitude: float = 1.0
    reference_kind: str = "manufactured"
    has_reference: bool = True

    @property
    def nu(self) -> float:
        return 1.0 / self.reynolds

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def grid(self, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(self.x_min, self.x_max, nx)
        y = np.linspace(self.y_min, self.y_max, ny)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
        return X, Y, xy

    def boundary_mask_np(self, xy: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        return (
            np.isclose(xy[:, 0], self.x_min, atol=tol)
            | np.isclose(xy[:, 0], self.x_max, atol=tol)
            | np.isclose(xy[:, 1], self.y_min, atol=tol)
            | np.isclose(xy[:, 1], self.y_max, atol=tol)
        )

    def exact_np(self, xy: np.ndarray) -> dict[str, np.ndarray]:
        x = torch.tensor(xy[:, 0:1], dtype=torch.float64)
        y = torch.tensor(xy[:, 1:2], dtype=torch.float64)
        return {k: v.detach().cpu().numpy().astype(float) for k, v in self._exact_torch_xy(x, y).items()}

    def exact_torch(self, xy: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._exact_torch_xy(xy[:, 0:1], xy[:, 1:2])

    def _exact_torch_xy(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError


@dataclass(frozen=True)
class PoiseuilleChannelFlow(RectangularBenchmarkBase):
    """Analytical pressure-driven channel with inflow/outflow boundaries."""

    reference_kind: str = "analytical"
    has_reference: bool = True

    def _exact_torch_xy(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        eta = (y - self.y_min) / self.height
        u = 4.0 * self.amplitude * eta * (1.0 - eta)
        v = torch.zeros_like(u)
        dpdx = -8.0 * self.nu * self.amplitude / (self.height * self.height)
        p = dpdx * (x - self.x_min)
        omega = -4.0 * self.amplitude * (1.0 - 2.0 * eta) / self.height
        px = torch.full_like(u, dpdx)
        py = torch.zeros_like(u)
        speed = torch.sqrt(u * u + v * v)
        return {"u": u, "v": v, "p": p, "omega": omega, "p_x": px, "p_y": py, "speed": speed}


@dataclass(frozen=True)
class DoubleVortexBoxFlow(RectangularBenchmarkBase):
    """Manufactured two-cell incompressible vortex field."""

    reference_kind: str = "manufactured"
    has_reference: bool = True

    def _exact_torch_xy(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        xi = (x - self.x_min) / self.width
        eta = (y - self.y_min) / self.height
        pi_t = torch.as_tensor(pi, dtype=x.dtype, device=x.device)
        a = self.amplitude
        u = a * torch.sin(2.0 * pi_t * xi).pow(2) * torch.sin(2.0 * pi_t * eta) * pi_t / self.height
        v = -a * torch.sin(4.0 * pi_t * xi) * torch.sin(pi_t * eta).pow(2) * pi_t / self.width
        p = 0.25 * a * (torch.cos(4.0 * pi_t * xi) + torch.cos(2.0 * pi_t * eta))
        px = -a * pi_t * torch.sin(4.0 * pi_t * xi) / self.width
        py = -0.5 * a * pi_t * torch.sin(2.0 * pi_t * eta) / self.height
        omega = torch.zeros_like(u)
        speed = torch.sqrt(u * u + v * v)
        return {"u": u, "v": v, "p": p, "omega": omega, "p_x": px, "p_y": py, "speed": speed}


@dataclass(frozen=True)
class BoundaryStressBoxFlow(RectangularBenchmarkBase):
    """Manufactured rectangular case with prescribed u/v boundary variation."""

    reference_kind: str = "manufactured"
    has_reference: bool = True

    def _exact_torch_xy(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        xi = (x - self.x_min) / self.width
        eta = (y - self.y_min) / self.height
        pi_t = torch.as_tensor(pi, dtype=x.dtype, device=x.device)
        a = self.amplitude
        u = a * (1.0 - eta) + 0.25 * a * torch.sin(pi_t * xi) * torch.sin(pi_t * eta)
        v = 0.25 * a * torch.sin(2.0 * pi_t * xi) * eta * (1.0 - eta)
        p = 0.1 * a * (1.0 - xi)
        px = torch.full_like(u, -0.1 * a / self.width)
        py = torch.zeros_like(u)
        omega = torch.zeros_like(u)
        speed = torch.sqrt(u * u + v * v)
        return {"u": u, "v": v, "p": p, "omega": omega, "p_x": px, "p_y": py, "speed": speed}


@dataclass(frozen=True)
class LidDrivenCavityQualitative(RectangularBenchmarkBase):
    """Boundary/residual-only lid-driven cavity benchmark without fake interior truth."""

    lid_velocity: float = 1.0
    reference_kind: str = "residual_only"
    has_reference: bool = False

    def _exact_torch_xy(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        tol = 1e-5
        u = torch.zeros_like(x)
        v = torch.zeros_like(x)
        top = torch.isclose(y, torch.full_like(y, self.y_max), atol=tol, rtol=0.0)
        u = torch.where(top, torch.full_like(u, self.lid_velocity), u)
        p = torch.zeros_like(x)
        omega = torch.zeros_like(x)
        px = torch.zeros_like(x)
        py = torch.zeros_like(x)
        speed = torch.sqrt(u * u + v * v)
        return {"u": u, "v": v, "p": p, "omega": omega, "p_x": px, "p_y": py, "speed": speed}
