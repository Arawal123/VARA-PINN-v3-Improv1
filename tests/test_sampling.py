from pathlib import Path
import sys

import numpy as np
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


def test_lid_cavity_boundary_sampler_focuses_lid_and_corners():
    device = torch.device("cpu")
    sampler = BoundarySampler((0, 1, 0, 1), device, 0)
    pts = sampler.sample_lid_cavity_numpy(200, lid_fraction=0.5, corner_fraction=0.25, corner_width=0.1)
    top_fraction = np.mean(np.isclose(pts[:, 1], 1.0))
    corner_zone = ((pts[:, 0] <= 0.1) | (pts[:, 0] >= 0.9)) & (pts[:, 1] >= 0.9)
    assert pts.shape == (200, 2)
    assert top_fraction > 0.45
    assert np.mean(corner_zone) > 0.10
