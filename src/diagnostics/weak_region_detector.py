"""Weak variable-region detection."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class WeakRegion:
    """Detected weak field-region pair."""

    patch_id: int
    variable: str
    severity: float
    confidence: float
    bounds: tuple[float, float, float, float, float | None, float | None]
    suggested_failure_type: str
    persistence: int = 1


class WeakRegionDetector:
    """Threshold, top-k, and persistence based weak-region detector."""

    def __init__(
        self,
        percentile_threshold: float = 80.0,
        top_k_per_variable: int = 2,
        min_active_patches: int = 1,
        max_active_patches: int = 8,
        persistence_cycles: int = 1,
    ) -> None:
        self.percentile_threshold = float(percentile_threshold)
        self.top_k_per_variable = int(top_k_per_variable)
        self.min_active_patches = int(min_active_patches)
        self.max_active_patches = int(max_active_patches)
        self.persistence_cycles = int(persistence_cycles)
        self._history: dict[tuple[str, int], int] = defaultdict(int)

    def detect(self, scores: np.ndarray, diagnostic_names: list[str], patch_grid: object) -> list[WeakRegion]:
        """Return active weak regions from S[j, r]."""
        if scores.size == 0:
            return []
        denom = scores.sum(axis=0, keepdims=True) + 1e-12
        confidence = scores / denom
        candidates: list[WeakRegion] = []
        active_keys: set[tuple[str, int]] = set()

        for j, name in enumerate(diagnostic_names):
            vals = scores[j]
            threshold = np.nanpercentile(vals, self.percentile_threshold)
            idx = np.argsort(vals)[::-1]
            selected = [int(pid) for pid in idx[: self.top_k_per_variable] if vals[pid] >= threshold]
            for pid in selected:
                key = (name, pid)
                active_keys.add(key)
                self._history[key] += 1
                if self._history[key] < self.persistence_cycles:
                    continue
                patch = patch_grid.get_patch(pid)
                candidates.append(
                    WeakRegion(
                        patch_id=pid,
                        variable=name,
                        severity=float(vals[pid]),
                        confidence=float(confidence[j, pid]),
                        bounds=patch.bounds,
                        suggested_failure_type=_failure_type(name, confidence[j, pid]),
                        persistence=self._history[key],
                    )
                )
        for key in list(self._history):
            if key not in active_keys:
                self._history[key] = 0

        candidates.sort(key=lambda wr: (wr.severity, wr.confidence), reverse=True)
        if len(candidates) < self.min_active_patches:
            flat = np.dstack(np.unravel_index(np.argsort(scores.ravel())[::-1], scores.shape))[0]
            for j, pid in flat:
                if len(candidates) >= self.min_active_patches:
                    break
                name = diagnostic_names[int(j)]
                if any(c.variable == name and c.patch_id == int(pid) for c in candidates):
                    continue
                patch = patch_grid.get_patch(int(pid))
                candidates.append(
                    WeakRegion(
                        patch_id=int(pid),
                        variable=name,
                        severity=float(scores[int(j), int(pid)]),
                        confidence=float(confidence[int(j), int(pid)]),
                        bounds=patch.bounds,
                        suggested_failure_type=_failure_type(name, confidence[int(j), int(pid)]),
                    )
                )
        return candidates[: self.max_active_patches]


def _failure_type(name: str, confidence: float) -> str:
    if "p_error" in name or "pressure" in name:
        root = "pressure"
    elif "omega" in name or "vorticity" in name:
        root = "vorticity"
    elif "u_error" in name or "v_error" in name or "speed" in name:
        root = "velocity"
    elif "continuity" in name:
        root = "divergence"
    elif "momentum" in name or "pde" in name:
        root = "residual"
    elif "boundary" in name:
        root = "boundary"
    else:
        root = "mixed"
    suffix = "dominant" if confidence >= 0.5 else "mixed"
    return f"{root}_{suffix}"

