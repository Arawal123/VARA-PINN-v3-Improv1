from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.physics.rectangular_benchmarks import LidDrivenCavityQualitative, PoiseuilleChannelFlow


def test_channel_reference_shapes():
    bench = PoiseuilleChannelFlow(x_max=2.0)
    xy = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
    ref = bench.exact_np(xy)
    assert ref["u"].shape == (3, 1)
    assert ref["v"].shape == (3, 1)
    assert ref["p"].shape == (3, 1)
    assert np.isclose(ref["u"][1, 0], 1.0)


def test_lid_cavity_boundary_reference_only():
    bench = LidDrivenCavityQualitative(lid_velocity=1.0)
    xy = torch.tensor([[0.5, 1.0], [0.5, 0.0]], dtype=torch.float32)
    ref = bench.exact_torch(xy)
    assert not bench.has_reference
    assert torch.isclose(ref["u"][0, 0], torch.tensor(1.0))
    assert torch.isclose(ref["u"][1, 0], torch.tensor(0.0))
