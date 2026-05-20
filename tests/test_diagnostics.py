from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diagnostics import DiagnosticMapBuilder
from src.models.mlp import MLP
from src.physics.kovasznay import KovasznayFlow


def test_diagnostic_builder_keys():
    flow = KovasznayFlow()
    model = MLP(2, 3, [8], input_lower=[flow.x_min, flow.y_min], input_upper=[flow.x_max, flow.y_max])
    _, _, xy = flow.grid(6, 5)
    maps = DiagnosticMapBuilder(model, flow, torch.device("cpu")).build(xy)
    for key in ["u_error", "p_error_mean_centered", "pressure_gradient_error", "pde_residual", "boundary_violation"]:
        assert key in maps
        assert maps[key].reshape(-1).shape[0] == xy.shape[0]
        assert np.isfinite(maps[key]).all()

