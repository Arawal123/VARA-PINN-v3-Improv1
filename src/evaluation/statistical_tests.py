"""Small statistical helpers for multi-seed comparisons."""

from __future__ import annotations

import numpy as np


def paired_mean_difference(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.asarray(a) - np.asarray(b)
    return {
        "mean_difference": float(np.mean(diff)),
        "std_difference": float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0,
        "n": int(diff.size),
    }

