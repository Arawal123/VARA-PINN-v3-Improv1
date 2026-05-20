from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.mlp import MLP
from src.physics.navier_stokes import navier_stokes_residuals


def test_residual_shapes():
    model = MLP(2, 3, [8, 8], input_lower=[-1, -1], input_upper=[1, 1])
    xy = torch.rand(10, 2)
    res = navier_stokes_residuals(model, xy, nu=0.01, steady=True)
    for key in ["f_c", "f_u", "f_v", "omega", "pde_residual"]:
        assert res[key].shape == (10, 1)
        assert torch.isfinite(res[key]).all()

