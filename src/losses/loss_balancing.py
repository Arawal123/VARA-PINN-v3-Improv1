"""Global loss balancing helpers."""

from __future__ import annotations


def clip_weights(weights: dict[str, float], min_value: float = 0.05, max_value: float = 50.0) -> dict[str, float]:
    """Clip scalar loss weights."""
    return {k: float(min(max(v, min_value), max_value)) for k, v in weights.items()}


def normalize_weight_sum(weights: dict[str, float], total: float | None = None) -> dict[str, float]:
    """Normalize weights to a target sum."""
    if total is None:
        total = float(len(weights))
    s = sum(float(v) for v in weights.values())
    if s <= 0:
        return weights
    return {k: float(v) * total / s for k, v in weights.items()}

