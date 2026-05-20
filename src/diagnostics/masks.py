"""Mask construction helpers."""

from __future__ import annotations

import numpy as np
import torch

from .patch_scores import PatchGrid


def build_numpy_masks(patch_ids: np.ndarray, active_patch_ids: list[int]) -> dict[int, np.ndarray]:
    """Create boolean masks for selected patches."""
    return {int(pid): patch_ids == int(pid) for pid in active_patch_ids}


def build_torch_masks(patch_ids: torch.Tensor, active_patch_ids: list[int]) -> dict[int, torch.Tensor]:
    """Create torch boolean masks for selected patches."""
    return {int(pid): patch_ids == int(pid) for pid in active_patch_ids}


def get_patch_mask(coords: np.ndarray | torch.Tensor, patch_grid: PatchGrid, patch_id: int) -> np.ndarray | torch.Tensor:
    """Return a boolean mask for points inside one patch."""
    if isinstance(coords, torch.Tensor):
        return patch_grid.mask_torch(coords, patch_id)
    return patch_grid.mask_numpy(coords, patch_id)
