"""Rectangular patch system and severity score tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .normalization import normalize_values, robust_aggregate


@dataclass(frozen=True)
class Patch:
    """Domain patch metadata."""

    patch_id: int
    ix: int
    iy: int
    it: int
    bounds: tuple[float, float, float, float, float | None, float | None]


class PatchGrid:
    """Axis-aligned rectangular patch grid."""

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        nx_patches: int = 4,
        ny_patches: int = 4,
        nt_patches: int = 1,
        t_bounds: tuple[float, float] | None = None,
    ) -> None:
        self.bounds = bounds
        self.nx_patches = int(nx_patches)
        self.ny_patches = int(ny_patches)
        self.nt_patches = int(nt_patches)
        self.t_bounds = t_bounds
        self.patches = self._build_patches()

    @property
    def num_patches(self) -> int:
        return self.nx_patches * self.ny_patches * self.nt_patches

    def _build_patches(self) -> list[Patch]:
        x0, x1, y0, y1 = self.bounds
        xs = np.linspace(x0, x1, self.nx_patches + 1)
        ys = np.linspace(y0, y1, self.ny_patches + 1)
        if self.t_bounds is None:
            ts = [None] * (self.nt_patches + 1)
        else:
            ts = list(np.linspace(self.t_bounds[0], self.t_bounds[1], self.nt_patches + 1))
        patches: list[Patch] = []
        pid = 0
        for it in range(self.nt_patches):
            for iy in range(self.ny_patches):
                for ix in range(self.nx_patches):
                    t0 = ts[it] if self.t_bounds is not None else None
                    t1 = ts[it + 1] if self.t_bounds is not None else None
                    patches.append(Patch(pid, ix, iy, it, (xs[ix], xs[ix + 1], ys[iy], ys[iy + 1], t0, t1)))
                    pid += 1
        return patches

    def get_patch(self, patch_id: int) -> Patch:
        return self.patches[int(patch_id)]

    def assign_numpy(self, coords: np.ndarray) -> np.ndarray:
        """Assign each coordinate to a patch id."""
        x0, x1, y0, y1 = self.bounds
        x = coords[:, 0]
        y = coords[:, 1]
        ix = np.floor((x - x0) / max(x1 - x0, 1e-12) * self.nx_patches).astype(int)
        iy = np.floor((y - y0) / max(y1 - y0, 1e-12) * self.ny_patches).astype(int)
        ix = np.clip(ix, 0, self.nx_patches - 1)
        iy = np.clip(iy, 0, self.ny_patches - 1)
        if self.nt_patches > 1 and coords.shape[1] >= 3 and self.t_bounds is not None:
            t0, t1 = self.t_bounds
            it = np.floor((coords[:, 2] - t0) / max(t1 - t0, 1e-12) * self.nt_patches).astype(int)
            it = np.clip(it, 0, self.nt_patches - 1)
        else:
            it = np.zeros_like(ix)
        return it * (self.nx_patches * self.ny_patches) + iy * self.nx_patches + ix

    def assign_torch(self, coords: torch.Tensor) -> torch.Tensor:
        """Torch patch assignment for masks in training loops."""
        x0, x1, y0, y1 = self.bounds
        x = coords[:, 0]
        y = coords[:, 1]
        ix = torch.floor((x - x0) / max(x1 - x0, 1e-12) * self.nx_patches).long()
        iy = torch.floor((y - y0) / max(y1 - y0, 1e-12) * self.ny_patches).long()
        ix = torch.clamp(ix, 0, self.nx_patches - 1)
        iy = torch.clamp(iy, 0, self.ny_patches - 1)
        if self.nt_patches > 1 and coords.shape[1] >= 3 and self.t_bounds is not None:
            t0, t1 = self.t_bounds
            it = torch.floor((coords[:, 2] - t0) / max(t1 - t0, 1e-12) * self.nt_patches).long()
            it = torch.clamp(it, 0, self.nt_patches - 1)
        else:
            it = torch.zeros_like(ix)
        return it * (self.nx_patches * self.ny_patches) + iy * self.nx_patches + ix

    def mask_numpy(self, coords: np.ndarray, patch_id: int) -> np.ndarray:
        return self.assign_numpy(coords) == int(patch_id)

    def mask_torch(self, coords: torch.Tensor, patch_id: int) -> torch.Tensor:
        return self.assign_torch(coords) == int(patch_id)


class PatchScorer:
    """Compute S[j, r] severity scores."""

    def __init__(
        self,
        patch_grid: PatchGrid,
        diagnostics: Iterable[str] | None = None,
        aggregation: str = "mean",
        normalization: str = "percentile",
        percentile: float = 90.0,
        ema_rho: float = 0.8,
    ) -> None:
        self.patch_grid = patch_grid
        self.diagnostics = list(diagnostics) if diagnostics is not None else None
        self.aggregation = aggregation
        self.normalization = normalization
        self.percentile = percentile
        self.ema_rho = float(ema_rho)
        self.previous: np.ndarray | None = None
        self.diagnostic_names: list[str] = []
        self.last_raw_scores: np.ndarray | None = None
        self.last_normalized_scores: np.ndarray | None = None

    def compute(
        self,
        diagnostic_maps: dict[str, np.ndarray],
        coords: np.ndarray,
        update_ema: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """Return severity tensor with shape [num_diagnostics, num_patches]."""
        names = self.diagnostics or [
            k for k, v in diagnostic_maps.items() if isinstance(v, np.ndarray) and v.reshape(-1).shape[0] == coords.shape[0]
        ]
        names = [n for n in names if n in diagnostic_maps]
        patch_ids = self.patch_grid.assign_numpy(coords)
        scores = np.zeros((len(names), self.patch_grid.num_patches), dtype=float)
        for j, name in enumerate(names):
            vals = np.asarray(diagnostic_maps[name]).reshape(-1)
            for pid in range(self.patch_grid.num_patches):
                scores[j, pid] = robust_aggregate(vals[patch_ids == pid], self.aggregation, self.percentile)
        raw_scores = scores.copy()
        for j in range(len(names)):
            scores[j] = normalize_values(scores[j], self.normalization)
        if update_ema:
            if self.previous is None or self.previous.shape != scores.shape:
                smoothed = scores
            else:
                smoothed = self.ema_rho * self.previous + (1.0 - self.ema_rho) * scores
            self.previous = smoothed
        else:
            smoothed = scores
        self.diagnostic_names = names
        self.last_raw_scores = raw_scores
        self.last_normalized_scores = smoothed
        return smoothed, names
