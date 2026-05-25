from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import StreamfunctionMLP, build_mlp_from_config
from src.physics.navier_stokes import navier_stokes_residuals


def test_streamfunction_model_returns_velocity_pressure_under_no_grad():
    model = StreamfunctionMLP(hidden_layers=[8], input_lower=[0, 0], input_upper=[1, 1])
    xy = torch.rand(6, 2)
    with torch.no_grad():
        out = model(xy)
    assert out.shape == (6, 3)
    assert torch.isfinite(out).all()


def test_streamfunction_model_is_nearly_divergence_free():
    model = StreamfunctionMLP(hidden_layers=[8], input_lower=[0, 0], input_upper=[1, 1])
    xy = torch.rand(8, 2)
    res = navier_stokes_residuals(model, xy, nu=0.01, steady=True)
    assert torch.max(torch.abs(res["f_c"])).detach().item() < 1.0e-4


def test_streamfunction_builder_from_config():
    cfg = {
        "benchmark_params": {"lid_velocity": 1.0},
        "model": {
            "type": "streamfunction",
            "input_dim": 2,
            "hidden_layers": [8],
            "activation": "tanh",
            "boundary_transform": True,
        },
    }
    model = build_mlp_from_config(cfg, (0, 1, 0, 1))
    assert isinstance(model, StreamfunctionMLP)

