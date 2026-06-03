from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_field_model_from_config
from src.physics.navier_stokes import navier_stokes_residuals


def test_streamfunction_velocity_is_autograd_divergence_free():
    config = {
        "model": {
            "kind": "streamfunction_p",
            "input_dim": 2,
            "hidden_layers": [16, 16],
            "activation": "tanh",
        }
    }
    model = build_field_model_from_config(config, (0.0, 1.0, 0.0, 1.0))
    coords = torch.rand(24, 2)
    residuals = navier_stokes_residuals(model, coords, nu=0.01, steady=True)
    assert float(torch.max(torch.abs(residuals["f_c"])).detach()) < 1e-5


def test_streamfunction_inference_works_under_no_grad():
    config = {
        "model": {
            "kind": "streamfunction_p",
            "input_dim": 2,
            "hidden_layers": [8],
            "activation": "tanh",
        }
    }
    model = build_field_model_from_config(config, (0.0, 1.0, 0.0, 1.0))
    with torch.no_grad():
        pred = model(torch.rand(5, 2))
    assert pred.shape == (5, 3)
    assert torch.isfinite(pred).all()
