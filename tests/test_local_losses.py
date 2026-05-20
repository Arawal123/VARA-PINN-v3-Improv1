from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diagnostics import PatchGrid
from src.losses.local_losses import compute_local_weighted_loss


def test_local_loss_no_nan_on_empty_mask():
    grid = PatchGrid(bounds=(0, 1, 0, 1), nx_patches=2, ny_patches=2)
    pointwise = {"u": torch.ones(4, 1)}
    batch = {"xy_data": torch.tensor([[0.1, 0.1], [0.2, 0.1], [0.1, 0.2], [0.2, 0.2]])}
    loss, logs = compute_local_weighted_loss(pointwise, batch, grid, {"u": {3: 10.0}})
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert "local/u/patch_3" in logs


def test_local_loss_affects_only_selected_patch():
    grid = PatchGrid(bounds=(0, 1, 0, 1), nx_patches=2, ny_patches=2)
    pointwise = {"u": torch.tensor([[1.0], [2.0], [10.0], [20.0]])}
    batch = {"xy_data": torch.tensor([[0.1, 0.1], [0.2, 0.1], [0.9, 0.9], [0.8, 0.9]])}
    loss, logs = compute_local_weighted_loss(pointwise, batch, grid, {"u": {0: 1.0}})
    assert torch.isclose(loss, torch.tensor(2.5))
    assert logs["local/u/patch_0"] == 2.5
