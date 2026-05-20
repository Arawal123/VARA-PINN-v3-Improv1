"""LBFGS closure helper."""

from __future__ import annotations

import torch


def make_lbfgs_closure(optimizer: torch.optim.Optimizer, loss_fn):
    """Create an LBFGS closure from a zero-argument loss function."""

    def closure():
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    return closure

