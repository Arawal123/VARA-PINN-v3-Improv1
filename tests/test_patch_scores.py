from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diagnostics import PatchGrid, PatchScorer
from src.diagnostics.masks import get_patch_mask


def test_patch_assignment_and_scores():
    grid = PatchGrid(bounds=(0, 1, 0, 1), nx_patches=2, ny_patches=2)
    coords = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
    ids = grid.assign_numpy(coords)
    assert set(ids.tolist()) == {0, 1, 2, 3}
    maps = {"u_error": np.array([[1.0], [2.0], [3.0], [4.0]])}
    scorer = PatchScorer(grid, diagnostics=["u_error"], normalization="none")
    scores, names = scorer.compute(maps, coords)
    assert names == ["u_error"]
    assert scores.shape == (1, 4)
    assert scores[0, 3] == 4.0
    assert scorer.last_raw_scores is not None


def test_patch_mask_correctness():
    grid = PatchGrid(bounds=(0, 1, 0, 1), nx_patches=2, ny_patches=2)
    coords = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
    mask = get_patch_mask(coords, grid, 2)
    assert mask.tolist() == [False, False, True, False]
