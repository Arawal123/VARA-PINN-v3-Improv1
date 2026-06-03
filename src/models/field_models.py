"""Field-model builders for direct and hard-divergence PINNs."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from src.models.fourier_features import FourierFeatures
from src.models.mlp import MLP


class FourierFeatureMLP(nn.Module):
    """MLP fed by Fourier features of normalized coordinates."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_layers: Iterable[int],
        activation: str,
        input_lower: Iterable[float],
        input_upper: Iterable[float],
        num_features: int = 64,
        scale: float = 5.0,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.include_input = bool(include_input)
        self.register_buffer("input_lower", torch.tensor(list(input_lower), dtype=torch.float32))
        self.register_buffer("input_upper", torch.tensor(list(input_upper), dtype=torch.float32))
        self.features = FourierFeatures(in_dim, num_features=num_features, scale=scale)
        encoded_dim = 2 * int(num_features) + (in_dim if include_input else 0)
        self.head = MLP(
            encoded_dim,
            out_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            input_lower=[-1.0] * encoded_dim,
            input_upper=[1.0] * encoded_dim,
        )

    def normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        lower = self.input_lower.to(device=x.device, dtype=x.dtype)
        upper = self.input_upper.to(device=x.device, dtype=x.dtype)
        return 2.0 * (x - lower) / (upper - lower).clamp_min(1e-12) - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.normalize_inputs(x)
        encoded = self.features(z)
        if self.include_input:
            encoded = torch.cat([z, encoded], dim=-1)
        return self.head(encoded)


class StreamfunctionPressureModel(nn.Module):
    """Hard-divergence model: raw network outputs psi,p; velocity is dpsi/dy,-dpsi/dx."""

    field_kind = "streamfunction_p"

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        grad_enabled = torch.is_grad_enabled()
        if grad_enabled:
            coords = xy if xy.requires_grad else xy.detach().clone().requires_grad_(True)
            raw = self.backbone(coords)
            psi = raw[:, 0:1]
            p = raw[:, 1:2]
            grad_psi = torch.autograd.grad(
                psi,
                coords,
                grad_outputs=torch.ones_like(psi),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            u = grad_psi[:, 1:2]
            v = -grad_psi[:, 0:1]
            return torch.cat([u, v, p], dim=1)

        with torch.enable_grad():
            coords = xy.detach().clone().requires_grad_(True)
            raw = self.backbone(coords)
            psi = raw[:, 0:1]
            p = raw[:, 1:2]
            grad_psi = torch.autograd.grad(
                psi,
                coords,
                grad_outputs=torch.ones_like(psi),
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
            )[0]
            u = grad_psi[:, 1:2]
            v = -grad_psi[:, 0:1]
            return torch.cat([u, v, p], dim=1)

    def raw(self, xy: torch.Tensor) -> torch.Tensor:
        """Return raw psi,p for diagnostics or checkpoint inspection."""
        return self.backbone(xy)


def build_field_model_from_config(config: dict, bounds: tuple[float, float, float, float]) -> nn.Module:
    """Build a field model from config, preserving direct u,v,p as the default."""
    model_cfg = config.get("model", {})
    x0, x1, y0, y1 = bounds
    in_dim = int(model_cfg.get("input_dim", 2))
    lower = [x0, y0] if in_dim == 2 else [x0, y0, model_cfg.get("t_min", 0.0)]
    upper = [x1, y1] if in_dim == 2 else [x1, y1, model_cfg.get("t_max", 1.0)]
    kind = str(model_cfg.get("kind", "direct_uvp")).lower()
    out_dim = 2 if kind == "streamfunction_p" else int(model_cfg.get("output_dim", 3))
    hidden_layers = model_cfg.get("hidden_layers", [96, 96, 96, 96])
    activation = model_cfg.get("activation", "tanh")
    fourier_cfg = model_cfg.get("fourier_features", {})
    if bool(fourier_cfg.get("enabled", False)):
        backbone: nn.Module = FourierFeatureMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            input_lower=lower,
            input_upper=upper,
            num_features=int(fourier_cfg.get("num_features", 64)),
            scale=float(fourier_cfg.get("scale", 5.0)),
            include_input=bool(fourier_cfg.get("include_input", True)),
        )
    else:
        backbone = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            input_lower=lower,
            input_upper=upper,
        )
    if kind == "direct_uvp":
        setattr(backbone, "field_kind", "direct_uvp")
        return backbone
    if kind == "streamfunction_p":
        return StreamfunctionPressureModel(backbone)
    raise ValueError(f"Unsupported model kind: {kind}. Use direct_uvp or streamfunction_p.")
