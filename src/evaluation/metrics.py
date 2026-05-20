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
    ref = benchmark.exact_np(coords_np)

    u = pred[:, 0:1].detach().cpu().numpy()
    v = pred[:, 1:2].detach().cpu().numpy()
    p = pred[:, 2:3].detach().cpu().numpy()
    omega = residuals["omega"].detach().cpu().numpy()
    p_c = center_pressure(p)
    p_ref_c = center_pressure(ref["p"])
    speed = np.sqrt(u * u + v * v)
    p_grad_err = np.sqrt(
        (residuals["p_x"].detach().cpu().numpy() - ref.get("p_x", 0.0)) ** 2
        + (residuals["p_y"].detach().cpu().numpy() - ref.get("p_y", 0.0)) ** 2
    )
    pde = residuals["pde_residual"].detach().cpu().numpy()
    div = np.abs(residuals["f_c"].detach().cpu().numpy())
    return {
        "u_rel_l2": relative_l2(u, ref["u"]),
        "v_rel_l2": relative_l2(v, ref["v"]),
        "p_rel_l2_centered": relative_l2(p_c, p_ref_c),
        "speed_rel_l2": relative_l2(speed, ref.get("speed", np.sqrt(ref["u"] ** 2 + ref["v"] ** 2))),
        "omega_rel_l2": relative_l2(omega, ref["omega"]),
        "pressure_gradient_error": float(np.mean(p_grad_err)),
        "divergence_norm": float(np.mean(div)),
        "pde_residual_mean": float(np.mean(pde)),
        "pde_residual_max": float(np.max(pde)),
        "wall_clock_eval_sec": time.time() - start,
        "num_eval_points": int(coords_np.shape[0]),
    }

