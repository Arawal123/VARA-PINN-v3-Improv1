"""Uniform domain sampling."""

from __future__ import annotations

import numpy as np
import torch


class UniformSampler:
    """Uniform interior sampler."""

    def __init__(self, bounds: tuple[float, float, float, float], device: torch.device, seed: int | None = None) -> None:
        self.bounds = bounds
        self.device = device
        self.rng = np.random.default_rng(seed)

    def sample_numpy(self, n: int) -> np.ndarray:
        x0, x1, y0, y1 = self.bounds
        return np.column_stack([self.rng.uniform(x0, x1, n), self.rng.uniform(y0, y1, n)])

    def sample(self, n: int) -> torch.Tensor:
        return torch.tensor(self.sample_numpy(n), dtype=torch.float32, device=self.device)

