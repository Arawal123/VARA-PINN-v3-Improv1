"""Physics operators and benchmark definitions."""

from .kovasznay import KovasznayFlow
from .navier_stokes import gradients, navier_stokes_residuals

__all__ = ["KovasznayFlow", "gradients", "navier_stokes_residuals"]

