"""Benchmark evaluator wrapper."""

from __future__ import annotations

from typing import Any

import torch

from .metrics import evaluate_on_grid


class BenchmarkEvaluator:
    def __init__(self, benchmark: Any, device: torch.device, steady: bool = True) -> None:
        self.benchmark = benchmark
        self.device = device
        self.steady = steady

    def evaluate(self, model: torch.nn.Module, coords_np) -> dict[str, float]:
        return evaluate_on_grid(model, self.benchmark, coords_np, self.device, self.steady)

