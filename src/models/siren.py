"""SIREN layers for oscillatory PINN alternatives."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """Linear layer followed by sine activation."""

    def __init__(self, in_features: int, out_features: int, omega0: float = 30.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega0 = omega0
        bound = 1.0 / in_features
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(x))


class SIREN(nn.Module):
    """Small SIREN network."""

    def __init__(self, in_dim: int = 2, out_dim: int = 3, hidden_dim: int = 96, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [SineLayer(in_dim, hidden_dim)]
        layers.extend(SineLayer(hidden_dim, hidden_dim, omega0=1.0) for _ in range(depth - 1))
        final = nn.Linear(hidden_dim, out_dim)
        nn.init.uniform_(final.weight, -math.sqrt(6 / hidden_dim) / 30.0, math.sqrt(6 / hidden_dim) / 30.0)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

