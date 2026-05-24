"""Physics operators and benchmark definitions."""

from .kovasznay import KovasznayFlow
from .navier_stokes import gradients, navier_stokes_residuals
from .rectangular_benchmarks import (
    BoundaryStressBoxFlow,
    DoubleVortexBoxFlow,
    LidDrivenCavityQualitative,
    PoiseuilleChannelFlow,
)

__all__ = [
    "KovasznayFlow",
    "PoiseuilleChannelFlow",
    "DoubleVortexBoxFlow",
    "BoundaryStressBoxFlow",
    "LidDrivenCavityQualitative",
    "gradients",
    "navier_stokes_residuals",
]

