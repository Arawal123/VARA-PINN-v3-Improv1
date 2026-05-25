"""Pointwise diagnostic map builder."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.physics.kovasznay import center_pressure
from src.physics.navier_stokes import navier_stokes_residuals
from src.physics.vorticity import vorticity_transport_residual


class DiagnosticMapBuilder:
    """Build pointwise diagnostic arrays for VARA diagnosis."""

    def __init__(
        self,
        model: torch.nn.Module,
        benchmark: Any,
        device: torch.device,
        steady: bool = True,
    ) -> None:
        self.model = model
        self.benchmark = benchmark
        self.device = device
        self.steady = steady

    def build(
        self,
        coords_np: np.ndarray,
        mode: str = "full_reference",
        boundary_coords_np: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute diagnostics in full_reference, sparse_data, or residual_only mode."""
        mode = mode.lower()
        coords = torch.tensor(coords_np, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(coords)
        residuals = navier_stokes_residuals(self.model, coords, nu=self.benchmark.nu, steady=self.steady)

        u_pred = pred[:, 0:1].detach().cpu().numpy()
        v_pred = pred[:, 1:2].detach().cpu().numpy()
        p_pred = pred[:, 2:3].detach().cpu().numpy()
        omega_pred = residuals["omega"].detach().cpu().numpy()
        p_x = residuals["p_x"].detach().cpu().numpy()
        p_y = residuals["p_y"].detach().cpu().numpy()
        speed_pred = np.sqrt(u_pred * u_pred + v_pred * v_pred)

        maps: dict[str, np.ndarray] = {
            "u_pred": u_pred,
            "v_pred": v_pred,
            "p_pred": center_pressure(p_pred),
            "omega_pred": omega_pred,
            "speed_pred": speed_pred,
            "continuity_residual": np.abs(residuals["f_c"].detach().cpu().numpy()),
            "momentum_u_residual": np.abs(residuals["f_u"].detach().cpu().numpy()),
            "momentum_v_residual": np.abs(residuals["f_v"].detach().cpu().numpy()),
            "pde_residual": residuals["pde_residual"].detach().cpu().numpy(),
        }

        if not self.steady:
            vt = vorticity_transport_residual(self.model, coords, nu=self.benchmark.nu, steady=False)
            maps["vorticity_transport_residual"] = np.abs(vt.detach().cpu().numpy())

        has_reference = bool(getattr(self.benchmark, "has_reference", True))
        if mode in {"full_reference", "sparse_data"} and has_reference:
            ref = self.benchmark.exact_np(coords_np)
            p_ref_c = center_pressure(ref["p"])
            p_pred_c = center_pressure(p_pred)
            p_grad_err = np.sqrt((p_x - ref.get("p_x", 0.0)) ** 2 + (p_y - ref.get("p_y", 0.0)) ** 2)
            maps.update(
                {
                    "u_ref": ref["u"],
                    "v_ref": ref["v"],
                    "p_ref": p_ref_c,
                    "omega_ref": ref["omega"],
                    "speed_ref": ref.get("speed", np.sqrt(ref["u"] ** 2 + ref["v"] ** 2)),
                    "u_error": np.abs(u_pred - ref["u"]),
                    "v_error": np.abs(v_pred - ref["v"]),
                    "p_error_mean_centered": np.abs(p_pred_c - p_ref_c),
                    "pressure_gradient_error": np.abs(p_grad_err),
                    "speed_error": np.abs(speed_pred - ref.get("speed", np.sqrt(ref["u"] ** 2 + ref["v"] ** 2))),
                    "omega_error": np.abs(omega_pred - ref["omega"]),
                }
            )
        else:
            zeros = np.zeros_like(u_pred)
            maps.update(
                {
                    "u_error": zeros,
                    "v_error": zeros,
                    "p_error_mean_centered": zeros,
                    "pressure_gradient_error": zeros,
                    "speed_error": zeros,
                    "omega_error": zeros,
                }
            )

        maps["aggregate_pde_residual"] = maps["pde_residual"]
        maps["boundary_violation"] = self._boundary_violation(coords_np, boundary_coords_np)
        maps.update(self._profile_errors(coords_np, u_pred, v_pred))
        return maps

    def _profile_errors(self, coords_np: np.ndarray, u_pred: np.ndarray, v_pred: np.ndarray) -> dict[str, np.ndarray]:
        zeros = np.zeros((coords_np.shape[0], 1), dtype=float)
        out = {"u_profile_error": zeros.copy(), "v_profile_error": zeros.copy(), "profile_error": zeros.copy()}
        if not bool(getattr(self.benchmark, "has_profile_reference", False)):
            return out
        profile = self.benchmark.profile_reference_np()
        if "u_xy" in profile and "u_ref" in profile:
            xy_ref = np.asarray(profile["u_xy"], dtype=float)
            values = np.asarray(profile["u_ref"], dtype=float).reshape(-1)
            order = np.argsort(xy_ref[:, 1])
            interp = np.interp(coords_np[:, 1], xy_ref[order, 1], values[order])
            center = float(np.mean(xy_ref[:, 0]))
            sigma = max((self.benchmark.bounds[1] - self.benchmark.bounds[0]) / 8.0, 1e-6)
            weight = np.exp(-((coords_np[:, 0] - center) / sigma) ** 2)
            out["u_profile_error"][:, 0] = np.abs(u_pred[:, 0] - interp) * weight
        if "v_xy" in profile and "v_ref" in profile:
            xy_ref = np.asarray(profile["v_xy"], dtype=float)
            values = np.asarray(profile["v_ref"], dtype=float).reshape(-1)
            order = np.argsort(xy_ref[:, 0])
            interp = np.interp(coords_np[:, 0], xy_ref[order, 0], values[order])
            center = float(np.mean(xy_ref[:, 1]))
            sigma = max((self.benchmark.bounds[3] - self.benchmark.bounds[2]) / 8.0, 1e-6)
            weight = np.exp(-((coords_np[:, 1] - center) / sigma) ** 2)
            out["v_profile_error"][:, 0] = np.abs(v_pred[:, 0] - interp) * weight
        out["profile_error"] = out["u_profile_error"] + out["v_profile_error"]
        return out

    def _boundary_violation(self, coords_np: np.ndarray, boundary_coords_np: np.ndarray | None) -> np.ndarray:
        x0, x1, y0, y1 = self.benchmark.bounds
        tol = 1e-6
        boundary_mask = (
            np.isclose(coords_np[:, 0], x0, atol=tol)
            | np.isclose(coords_np[:, 0], x1, atol=tol)
            | np.isclose(coords_np[:, 1], y0, atol=tol)
            | np.isclose(coords_np[:, 1], y1, atol=tol)
        )
        out = np.zeros((coords_np.shape[0], 1), dtype=float)
        if not np.any(boundary_mask):
            return out
        coords = torch.tensor(coords_np[boundary_mask], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self.model(coords)
            ref = self.benchmark.exact_torch(coords)
            viol = torch.sqrt((pred[:, 0:1] - ref["u"]).pow(2) + (pred[:, 1:2] - ref["v"]).pow(2))
        out[boundary_mask] = viol.cpu().numpy()
        return out
