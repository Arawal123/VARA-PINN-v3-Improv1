"""Reference-data utilities for lid-driven cavity benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "references" / "lid_driven_cavity"
FULL_FIELD_DIR = REFERENCE_DIR / "full_field"


@dataclass(frozen=True)
class CavityProfileReference:
    """Centerline profile data for lid-driven cavity validation."""

    reynolds: float
    source: str
    u_profile: pd.DataFrame
    v_profile: pd.DataFrame

    @property
    def has_u(self) -> bool:
        return not self.u_profile.empty

    @property
    def has_v(self) -> bool:
        return not self.v_profile.empty


def load_lid_cavity_profile_reference(
    reference: str,
    reynolds: float,
    reference_path: str | Path | None = None,
) -> CavityProfileReference:
    """Load built-in Ghia or external centerline data for one Reynolds number."""
    reference = (reference or "none").lower()
    if reference == "none":
        return CavityProfileReference(reynolds, "none", pd.DataFrame(), pd.DataFrame())
    if reference == "ghia":
        return _load_ghia(reynolds)
    if reference == "external":
        if reference_path is None:
            raise ValueError("--reference external requires --reference_path.")
        return _load_external_profile(Path(reference_path), reynolds)
    raise ValueError(f"Unknown cavity reference '{reference}'. Use ghia, external, or none.")


def load_full_field_reference(path: str | Path) -> dict[str, np.ndarray]:
    """Load optional structured full-field CFD reference from CSV/NPZ/NPY."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Full-field cavity reference not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return _columns_to_field_dict(df.to_dict("list"), path)
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        return _columns_to_field_dict({k: data[k] for k in data.files}, path)
    if getattr(data, "dtype", None) is not None and data.dtype.names:
        return _columns_to_field_dict({name: data[name] for name in data.dtype.names}, path)
    if data.shape[-1] < 4:
        raise ValueError("Full-field NPY reference must have columns x,y,u,v and optional p,omega.")
    names = ["x", "y", "u", "v", "p", "omega"]
    return _columns_to_field_dict({name: data[..., i].reshape(-1) for i, name in enumerate(names[: data.shape[-1]])}, path)


def load_or_generate_full_field_reference(
    reynolds: float,
    path: str | Path | None = None,
    nx: int = 65,
    ny: int | None = None,
    max_iter: int = 4000,
    tolerance: float = 1e-6,
    force: bool = False,
) -> tuple[Path, dict[str, float]]:
    """Return an external full-field reference path or generate a deterministic CFD reference."""
    if path not in {None, "", "auto"}:
        ref_path = Path(path)
        validation = validate_full_field_against_ghia(load_full_field_reference(ref_path), reynolds)
        return ref_path, validation
    ny = int(ny or nx)
    out_dir = FULL_FIELD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"re{int(reynolds):04d}_nx{int(nx)}_ny{int(ny)}.npz"
    validation_path = out_dir / f"re{int(reynolds):04d}_nx{int(nx)}_ny{int(ny)}_validation.json"
    if out_path.exists() and not force:
        validation = validate_full_field_against_ghia(load_full_field_reference(out_path), reynolds)
        if not validation_path.exists():
            validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        return out_path, validation
    reference, validation = generate_lid_cavity_full_field_reference(
        reynolds=float(reynolds),
        nx=int(nx),
        ny=int(ny),
        max_iter=int(max_iter),
        tolerance=float(tolerance),
    )
    np.savez(out_path, **reference)
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return out_path, validation


def generate_lid_cavity_full_field_reference(
    reynolds: float,
    nx: int = 65,
    ny: int | None = None,
    max_iter: int = 4000,
    tolerance: float = 1e-6,
    lid_velocity: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Generate a deterministic vorticity-streamfunction finite-difference reference."""
    if int(round(reynolds)) not in {100, 400, 1000}:
        raise ValueError("Generated cavity references currently support Re=100, Re=400, and Re=1000.")
    ny = int(ny or nx)
    if nx < 9 or ny < 9:
        raise ValueError("Generated cavity reference grid must be at least 9x9.")
    x = np.linspace(0.0, 1.0, int(nx))
    y = np.linspace(0.0, 1.0, int(ny))
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    nu = 1.0 / float(reynolds)
    psi = np.zeros((ny, nx), dtype=float)
    omega = np.zeros((ny, nx), dtype=float)
    dt_diff = 0.20 / max(nu * (1.0 / dx**2 + 1.0 / dy**2), 1e-12)
    dt = min(0.0025, dt_diff, 0.25 * min(dx, dy) / max(float(lid_velocity), 1e-12))
    last_delta = float("inf")
    for _ in range(int(max_iter)):
        _apply_cavity_vorticity_boundary(psi, omega, dx, dy, lid_velocity)
        _solve_streamfunction_poisson(psi, omega, dx, dy, iterations=35, omega_relax=1.0)
        u, v = _velocity_from_streamfunction(psi, dx, dy, lid_velocity)
        omega_old = omega.copy()
        ox = (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2.0 * dx)
        oy = (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2.0 * dy)
        lap = (
            (omega[1:-1, 2:] - 2.0 * omega[1:-1, 1:-1] + omega[1:-1, :-2]) / dx**2
            + (omega[2:, 1:-1] - 2.0 * omega[1:-1, 1:-1] + omega[:-2, 1:-1]) / dy**2
        )
        omega[1:-1, 1:-1] = omega[1:-1, 1:-1] + dt * (
            -u[1:-1, 1:-1] * ox - v[1:-1, 1:-1] * oy + nu * lap
        )
        last_delta = float(np.max(np.abs(omega - omega_old)))
        if last_delta < float(tolerance):
            break
    _apply_cavity_vorticity_boundary(psi, omega, dx, dy, lid_velocity)
    _solve_streamfunction_poisson(psi, omega, dx, dy, iterations=100, omega_relax=1.0)
    u, v = _velocity_from_streamfunction(psi, dx, dy, lid_velocity)
    X, Y = np.meshgrid(x, y)
    reference = {
        "x": X.reshape(-1),
        "y": Y.reshape(-1),
        "u": u.reshape(-1),
        "v": v.reshape(-1),
        "p": np.zeros(nx * ny, dtype=float),
        "omega": omega.reshape(-1),
    }
    validation = validate_full_field_against_ghia(reference, reynolds)
    validation["solver_last_delta"] = last_delta
    validation["solver_grid_nx"] = int(nx)
    validation["solver_grid_ny"] = int(ny)
    validation["solver_max_iter"] = int(max_iter)
    return reference, validation


def interpolate_full_field(reference: dict[str, np.ndarray], xy: np.ndarray) -> dict[str, np.ndarray]:
    """Bilinearly interpolate a structured full-field reference onto points."""
    x = np.asarray(reference["x"], dtype=float).reshape(-1)
    y = np.asarray(reference["y"], dtype=float).reshape(-1)
    xu = np.unique(x)
    yu = np.unique(y)
    nx, ny = len(xu), len(yu)
    if nx * ny != x.size:
        raise ValueError("Full-field reference must be a structured tensor-product grid.")
    order = np.lexsort((x, y))
    out: dict[str, np.ndarray] = {}
    for name in ["u", "v", "p", "omega"]:
        values = np.asarray(reference.get(name, np.zeros_like(x)), dtype=float).reshape(-1)[order].reshape(ny, nx)
        out[name] = _interp2(xu, yu, values, xy[:, 0], xy[:, 1]).reshape(-1, 1)
    out["speed"] = np.sqrt(out["u"] ** 2 + out["v"] ** 2)
    out["p_x"] = np.zeros_like(out["u"])
    out["p_y"] = np.zeros_like(out["u"])
    return out


def validate_full_field_against_ghia(reference: dict[str, np.ndarray], reynolds: float) -> dict[str, float]:
    """Compare a full-field reference against built-in Ghia centerline profiles."""
    profile = load_lid_cavity_profile_reference("ghia", reynolds)
    out: dict[str, float] = {"ghia_reynolds": float(reynolds)}
    if profile.has_u:
        xy = np.column_stack([np.full(len(profile.u_profile), 0.5), profile.u_profile["y"].to_numpy(dtype=float)])
        interp = interpolate_full_field(reference, xy)["u"].reshape(-1)
        ref = profile.u_profile["u_ref"].to_numpy(dtype=float)
        out["ghia_u_centerline_rmse"] = float(np.sqrt(np.mean((interp - ref) ** 2)))
        out["ghia_u_centerline_max_abs"] = float(np.max(np.abs(interp - ref)))
        out["ghia_u_extrema_error"] = float(abs(np.min(interp) - np.min(ref)) + abs(np.max(interp) - np.max(ref)))
    if profile.has_v:
        xy = np.column_stack([profile.v_profile["x"].to_numpy(dtype=float), np.full(len(profile.v_profile), 0.5)])
        interp = interpolate_full_field(reference, xy)["v"].reshape(-1)
        ref = profile.v_profile["v_ref"].to_numpy(dtype=float)
        out["ghia_v_centerline_rmse"] = float(np.sqrt(np.mean((interp - ref) ** 2)))
        out["ghia_v_centerline_max_abs"] = float(np.max(np.abs(interp - ref)))
        out["ghia_v_extrema_error"] = float(abs(np.min(interp) - np.min(ref)) + abs(np.max(interp) - np.max(ref)))
    values = [v for k, v in out.items() if k.startswith("ghia_") and k.endswith(("rmse", "max_abs", "error"))]
    out["ghia_validation_score"] = float(np.sum(values)) if values else float("nan")
    return out


def _apply_cavity_vorticity_boundary(
    psi: np.ndarray,
    omega: np.ndarray,
    dx: float,
    dy: float,
    lid_velocity: float,
) -> None:
    omega[0, :] = -2.0 * psi[1, :] / dy**2
    omega[-1, :] = -2.0 * psi[-2, :] / dy**2 - 2.0 * lid_velocity / dy
    omega[:, 0] = -2.0 * psi[:, 1] / dx**2
    omega[:, -1] = -2.0 * psi[:, -2] / dx**2


def _solve_streamfunction_poisson(
    psi: np.ndarray,
    omega: np.ndarray,
    dx: float,
    dy: float,
    iterations: int,
    omega_relax: float = 1.5,
) -> None:
    denom = 2.0 * (dx * dx + dy * dy)
    for _ in range(int(iterations)):
        new_val = (
            dy * dy * (psi[1:-1, 2:] + psi[1:-1, :-2])
            + dx * dx * (psi[2:, 1:-1] + psi[:-2, 1:-1])
            + dx * dx * dy * dy * omega[1:-1, 1:-1]
        ) / denom
        psi[1:-1, 1:-1] = (1.0 - omega_relax) * psi[1:-1, 1:-1] + omega_relax * new_val
        psi[0, :] = 0.0
        psi[-1, :] = 0.0
        psi[:, 0] = 0.0
        psi[:, -1] = 0.0


def _velocity_from_streamfunction(
    psi: np.ndarray,
    dx: float,
    dy: float,
    lid_velocity: float,
) -> tuple[np.ndarray, np.ndarray]:
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * dy)
    v[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2.0 * dx)
    u[-1, :] = float(lid_velocity)
    u[0, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    v[0, :] = 0.0
    v[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    return u, v


def _load_ghia(reynolds: float) -> CavityProfileReference:
    u_path = REFERENCE_DIR / "ghia_1982_u_centerline.csv"
    v_path = REFERENCE_DIR / "ghia_1982_v_centerline.csv"
    u = _filter_re(pd.read_csv(u_path), reynolds, "u_ref", u_path)
    v = _filter_re(pd.read_csv(v_path), reynolds, "v_ref", v_path)
    return CavityProfileReference(float(reynolds), "ghia_1982", u, v)


def _load_external_profile(path: Path, reynolds: float) -> CavityProfileReference:
    if not path.exists():
        raise FileNotFoundError(f"External cavity profile reference not found: {path}")
    df = pd.read_csv(path)
    required = {"re", "x", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"External cavity profile CSV is missing columns: {sorted(missing)}")
    if "u_ref" not in df.columns and "v_ref" not in df.columns:
        raise ValueError("External cavity profile CSV must include u_ref and/or v_ref.")
    u = _filter_re(df.dropna(subset=["u_ref"]) if "u_ref" in df.columns else pd.DataFrame(), reynolds, "u_ref", path)
    v = _filter_re(df.dropna(subset=["v_ref"]) if "v_ref" in df.columns else pd.DataFrame(), reynolds, "v_ref", path)
    return CavityProfileReference(float(reynolds), str(path), u, v)


def _filter_re(df: pd.DataFrame, reynolds: float, value_col: str, source: Path) -> pd.DataFrame:
    if df.empty:
        return df
    required = {"re", "x", "y", value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    sub = df[np.isclose(df["re"].astype(float), float(reynolds))].copy()
    if sub.empty:
        available = sorted(float(v) for v in df["re"].dropna().unique())
        raise ValueError(f"No lid-driven cavity reference rows for Re={reynolds:g} in {source}. Available Re: {available}")
    return sub.sort_values(["y", "x"]).reset_index(drop=True)


def _columns_to_field_dict(columns: dict[str, Any], source: Path) -> dict[str, np.ndarray]:
    required = {"x", "y", "u", "v"}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Full-field reference {source} is missing columns/keys: {sorted(missing)}")
    out = {name: np.asarray(values, dtype=float).reshape(-1) for name, values in columns.items() if name in {"x", "y", "u", "v", "p", "omega"}}
    n = len(out["x"])
    if any(len(values) != n for values in out.values()):
        raise ValueError(f"Full-field reference {source} has inconsistent column lengths.")
    out.setdefault("p", np.zeros(n, dtype=float))
    out.setdefault("omega", np.zeros(n, dtype=float))
    return out


def _interp2(xu: np.ndarray, yu: np.ndarray, values: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
    xq = np.clip(xq, xu[0], xu[-1])
    yq = np.clip(yq, yu[0], yu[-1])
    ix = np.clip(np.searchsorted(xu, xq, side="right") - 1, 0, len(xu) - 2)
    iy = np.clip(np.searchsorted(yu, yq, side="right") - 1, 0, len(yu) - 2)
    x0, x1 = xu[ix], xu[ix + 1]
    y0, y1 = yu[iy], yu[iy + 1]
    tx = np.divide(xq - x0, x1 - x0, out=np.zeros_like(xq, dtype=float), where=(x1 != x0))
    ty = np.divide(yq - y0, y1 - y0, out=np.zeros_like(yq, dtype=float), where=(y1 != y0))
    v00 = values[iy, ix]
    v10 = values[iy, ix + 1]
    v01 = values[iy + 1, ix]
    v11 = values[iy + 1, ix + 1]
    return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11
