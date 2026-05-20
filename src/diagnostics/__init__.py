"""Diagnostics, patch scores, and weak-region detection."""

from .diagnostic_maps import DiagnosticMapBuilder
from .patch_scores import Patch, PatchGrid, PatchScorer
from .weak_region_detector import WeakRegion, WeakRegionDetector

__all__ = ["DiagnosticMapBuilder", "Patch", "PatchGrid", "PatchScorer", "WeakRegion", "WeakRegionDetector"]

