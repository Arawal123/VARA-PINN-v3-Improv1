"""Global model metrics."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from src.physics.kovasznay import center_pressure
from src.physics.navier_stokes import navier_stokes_residuals


def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def evaluate_on_grid(
    model: torch.nn.Module,
    benchmark: Any,
    coords_np: np.ndarray,
    device: torch.device,
    steady: bool = True,
) -> dict[str, float]:
    """Compute global evaluation metrics without using them for adaptation."""
    start = time.time()
    coords = torch.tensor(coords_np, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred = model(coords)
    residuals = navier_stokes_residuals(model, coords, nu=benchmark.nu, steady=steady)
    has_reference = bool(getattr(benchmark, "has_reference", True))
    ref = benchmark.exact_np(coords_np) if has_reference else None

    u = pred[:, 0:1].detach().cpu().numpy()
    v = pred[:, 1:2].detach().cpu().numpy()
    p = pred[:, 2:3].detach().cpu().numpy()
    omega = residuals["omega"].detach().cpu().numpy()
    p_c = center_pressure(p)
    speed = np.sqrt(u * u + v * v)
    pde = residuals["pde_residual"].detach().cpu().numpy()
    div = np.abs(residuals["f_c"].detach().cpu().numpy())
    metrics = {
        "u_rel_l2": float("nan"),
        "v_rel_l2": float("nan"),
        "p_rel_l2_centered": float("nan"),
        "speed_rel_l2": float("nan"),
        "omega_rel_l2": float("nan"),
        "pressure_gradient_error": float("nan"),
        "divergence_norm": float(np.mean(div)),
        "continuity_residual_mean": float(np.mean(np.abs(residuals["f_c"].detach().cpu().numpy()))),
        "momentum_residual_mean": float(
            np.mean(
                np.sqrt(
                    residuals["f_u"].detach().cpu().numpy() ** 2
                    + residuals["f_v"].detach().cpu().numpy() ** 2
                )
            )
        ),
        "pde_residual_mean": float(np.mean(pde)),
        "pde_residual_max": float(np.max(pde)),
        "boundary_condition_error": _boundary_error(model, benchmark, coords_np, device),
        "wall_clock_eval_sec": time.time() - start,
        "num_eval_points": int(coords_np.shape[0]),
    }
    if has_reference and ref is not None:
        p_ref_c = center_pressure(ref["p"])
        p_grad_err = np.sqrt(
            (residuals["p_x"].detach().cpu().numpy() - ref.get("p_x", 0.0)) ** 2
            + (residuals["p_y"].detach().cpu().numpy() - ref.get("p_y", 0.0)) ** 2
        )
        metrics.update(
            {
                "u_rel_l2": relative_l2(u, ref["u"]),
                "v_rel_l2": relative_l2(v, ref["v"]),
                "p_rel_l2_centered": relative_l2(p_c, p_ref_c),
                "speed_rel_l2": relative_l2(speed, ref.get("speed", np.sqrt(ref["u"] ** 2 + ref["v"] ** 2))),
                "omega_rel_l2": relative_l2(omega, ref["omega"]),
                "pressure_gradient_error": float(np.mean(p_grad_err)),
            }
        )
    return metrics


def _boundary_error(
    model: torch.nn.Module,
    benchmark: Any,
    coords_np: np.ndarray,
    device: torch.device,
) -> float:
    if not hasattr(benchmark, "boundary_mask_np"):
        return float("nan")
    mask = benchmark.boundary_mask_np(coords_np)
    if not np.any(mask):
        return float("nan")
    coords = torch.tensor(coords_np[mask], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(coords)
        ref = benchmark.exact_torch(coords)
        err = torch.sqrt((pred[:, 0:1] - ref["u"]).pow(2) + (pred[:, 1:2] - ref["v"]).pow(2))
    return float(torch.mean(err).detach().cpu())
