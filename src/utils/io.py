"""Experiment file IO."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


class JsonNumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays/scalars and dataclasses."""

    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON with scientific Python support."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=JsonNumpyEncoder)


def load_json(path: str | Path) -> Any:
    """Load JSON data."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

