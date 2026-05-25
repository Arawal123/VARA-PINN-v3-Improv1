"""Reference-data utilities for lid-driven cavity benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "references" / "lid_driven_cavity"


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
