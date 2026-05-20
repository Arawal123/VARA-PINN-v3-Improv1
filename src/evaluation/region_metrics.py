"""Regional metrics and controller improvement scores."""

from __future__ import annotations

import numpy as np


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(values.reshape(-1)[mask.reshape(-1)]))


def targeted_field_region_improvement(before: float, after: float, eps: float = 1e-12) -> float:
    return float((before - after) / (before + eps))


def spillover_degradation(before: np.ndarray, after: np.ndarray, target_j: int, target_r: int) -> float:
    degradation = np.maximum(after - before, 0.0)
    degradation[target_j, target_r] = 0.0
    return float(np.max(degradation))

