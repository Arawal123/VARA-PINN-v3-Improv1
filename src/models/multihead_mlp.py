"""Shared-trunk multi-head MLP."""

from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import MLP


class MultiHeadMLP(nn.Module):
    """Shared feature extractor with separate u, v, p heads."""

    def __init__(self, in_dim: int = 2, width: int = 96, depth: int = 4) -> None:
        super().__init__()
        self.trunk = MLP(in_dim, width, [width] * depth)
        self.u_head = nn.Linear(width, 1)
        self.v_head = nn.Linear(width, 1)
        self.p_head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk.normalize_inputs(x)
        for layer in self.trunk.layers[:-1]:
            z = self.trunk.activation(layer(z))
        features = self.trunk.layers[-1](z)
        return torch.cat([self.u_head(features), self.v_head(features), self.p_head(features)], dim=1)

