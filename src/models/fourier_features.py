"""Fourier feature positional encoding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Random Fourier feature map."""

    def __init__(self, in_dim: int, num_features: int = 64, scale: float = 5.0) -> None:
        super().__init__()
        B = torch.randn(in_dim, num_features) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * x @ self.B.to(device=x.device, dtype=x.dtype)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

