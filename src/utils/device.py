"""Device selection."""

from __future__ import annotations

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Resolve a device preference into a torch.device."""
    preference = preference.lower()
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preference)

