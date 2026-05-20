"""Baseline and ablation trainers."""

from __future__ import annotations

from src.training.vara_trainer import VARATrainer


class BaselineTrainer(VARATrainer):
    """Use the same trainer surface with controller behavior selected by mode."""

    def __init__(self, config: dict, mode: str = "vanilla_pinn") -> None:
        super().__init__(config, mode)

