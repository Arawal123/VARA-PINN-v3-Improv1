"""Taylor-Green vortex benchmark skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np


@dataclass(frozen=True)
class TaylorGreenVortex:
    """Analytical 2D Taylor-Green vortex reference."""

    reynolds: float = 100.0
    x_min: float = 0.0
    x_max: float = 2.0 * pi
    y_min: float = 0.0
    y_max: float = 2.0 * pi

    @property
    def nu(self) -> float:
        return 1.0 / self.reynolds

    def exact_np(self, xyt: np.ndarray) -> dict[str, np.ndarray]:
        x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
        decay = np.exp(-2.0 * self.nu * t)
        u = -np.cos(x) * np.sin(y) * decay
        v = np.sin(x) * np.cos(y) * decay
        p = -0.25 * (np.cos(2.0 * x) + np.cos(2.0 * y)) * decay * decay
        omega = 2.0 * np.sin(x) * np.sin(y) * decay
        return {"u": u, "v": v, "p": p, "omega": omega}

