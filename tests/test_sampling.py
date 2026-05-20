from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diagnostics import PatchGrid
from src.sampling import BoundarySampler, MixedAdaptiveSampler, UniformSampler


def test_sampler_point_counts():
    device = torch.device("cpu")
    bounds = (0, 1, 0, 1)
    grid = PatchGrid(bounds, 2, 2)
    assert UniformSampler(bounds, device, 0).sample(17).shape == (17, 2)
    assert BoundarySampler(bounds, device, 0).sample(18).shape == (18, 2)
    assert MixedAdaptiveSampler(bounds, grid, device, 0).sample_interior(19).shape == (19, 2)

