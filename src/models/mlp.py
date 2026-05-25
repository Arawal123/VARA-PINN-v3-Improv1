"""Configurable tanh MLP for velocity-pressure PINNs."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully connected MLP with optional affine input normalization."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_layers: Iterable[int] = (96, 96, 96, 96),
        activation: str = "tanh",
        input_lower: Iterable[float] | None = None,
        input_upper: Iterable[float] | None = None,
    ) -> None:
        super().__init__()
        sizes = [in_dim, *list(hidden_layers), out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )
        self.activation_name = activation
        self.activation = _activation(activation)
        if input_lower is None:
            input_lower = [-1.0] * in_dim
        if input_upper is None:
            input_upper = [1.0] * in_dim
        self.register_buffer("input_lower", torch.tensor(list(input_lower), dtype=torch.float32))
        self.register_buffer("input_upper", torch.tensor(list(input_upper), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        lower = self.input_lower.to(device=x.device, dtype=x.dtype)
        upper = self.input_upper.to(device=x.device, dtype=x.dtype)
        return 2.0 * (x - lower) / (upper - lower).clamp_min(1e-12) - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.normalize_inputs(x)
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))
        return self.layers[-1](z)


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp_from_config(config: dict, bounds: tuple[float, float, float, float]) -> nn.Module:
    """Build a velocity-pressure MLP from a config dictionary."""
    model_cfg = config.get("model", {})
    x0, x1, y0, y1 = bounds
    in_dim = int(model_cfg.get("input_dim", 2))
    lower = [x0, y0] if in_dim == 2 else [x0, y0, model_cfg.get("t_min", 0.0)]
    upper = [x1, y1] if in_dim == 2 else [x1, y1, model_cfg.get("t_max", 1.0)]
    model_type = str(model_cfg.get("type", model_cfg.get("name", "mlp"))).lower()
    if model_type in {"streamfunction", "streamfunction_mlp", "psi_p"}:
        from .streamfunction_mlp import StreamfunctionMLP

        bench_cfg = config.get("benchmark_params", {})
        return StreamfunctionMLP(
            in_dim=in_dim,
            hidden_layers=model_cfg.get("hidden_layers", [96, 96, 96, 96]),
            activation=model_cfg.get("activation", "tanh"),
            input_lower=lower,
            input_upper=upper,
            lid_velocity=float(bench_cfg.get("lid_velocity", model_cfg.get("lid_velocity", 1.0))),
            boundary_transform=bool(model_cfg.get("boundary_transform", True)),
            enforce_lid_profile=bool(model_cfg.get("enforce_lid_profile", False)),
            corner_eps=float(model_cfg.get("corner_eps", 0.08)),
            include_vorticity_transport_loss=bool(model_cfg.get("include_vorticity_transport_loss", False)),
        )
    return MLP(
        in_dim=in_dim,
        out_dim=int(model_cfg.get("output_dim", 3)),
        hidden_layers=model_cfg.get("hidden_layers", [96, 96, 96, 96]),
        activation=model_cfg.get("activation", "tanh"),
        input_lower=lower,
        input_upper=upper,
    )
