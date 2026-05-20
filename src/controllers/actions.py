"""Reversible local intervention actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingControlState:
    """Controller-modifiable training state."""

    global_weights: dict[str, float]
    local_weights: dict[str, dict[int, float]] = field(default_factory=dict)
    sampling_priorities: dict[int, float] = field(default_factory=dict)
    active_aux_losses: set[str] = field(default_factory=set)
    pressure_anchor_patches: dict[int, float] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "global_weights": dict(self.global_weights),
            "local_weights": {k: dict(v) for k, v in self.local_weights.items()},
            "sampling_priorities": dict(self.sampling_priorities),
            "active_aux_losses": set(self.active_aux_losses),
            "pressure_anchor_patches": dict(self.pressure_anchor_patches),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.global_weights = dict(snapshot["global_weights"])
        self.local_weights = {k: dict(v) for k, v in snapshot["local_weights"].items()}
        self.sampling_priorities = dict(snapshot["sampling_priorities"])
        self.active_aux_losses = set(snapshot["active_aux_losses"])
        self.pressure_anchor_patches = dict(snapshot["pressure_anchor_patches"])


class InterventionAction:
    """Base class for reversible interventions."""

    name = "NoOpAction"

    def __init__(self, patch_id: int | None = None, variable: str | None = None, strength: float = 1.0) -> None:
        self.patch_id = patch_id
        self.variable = variable
        self.strength = float(strength)
        self._snapshot: dict[str, Any] | None = None

    def apply(self, state: TrainingControlState) -> dict[str, Any]:
        self._snapshot = state.snapshot()
        return self.metadata()

    def rollback(self, state: TrainingControlState) -> None:
        if self._snapshot is not None:
            state.restore(self._snapshot)

    def metadata(self) -> dict[str, Any]:
        return {"action": self.name, "patch_id": self.patch_id, "variable": self.variable, "strength": self.strength}


def _bump_local(state: TrainingControlState, variable: str, patch_id: int, amount: float, clip: float = 30.0) -> None:
    state.local_weights.setdefault(variable, {})
    old = float(state.local_weights[variable].get(int(patch_id), 0.0))
    state.local_weights[variable][int(patch_id)] = min(old + float(amount), clip)


def cap_control_state(
    state: TrainingControlState,
    caps: dict[str, float] | None = None,
) -> None:
    """Clip global and local intervention weights after an action update."""
    caps = caps or {"u": 2.0, "v": 2.0, "p": 3.0, "omega": 2.0, "bc": 7.0}
    for name, cap in caps.items():
        if name in state.global_weights:
            state.global_weights[name] = min(float(state.global_weights[name]), float(cap))
        if name in state.local_weights:
            state.local_weights[name] = {
                int(pid): min(float(weight), float(cap)) for pid, weight in state.local_weights[name].items()
            }


class IncreaseLocalVelocityLoss(InterventionAction):
    name = "IncreaseLocalVelocityLoss"

    def apply(self, state: TrainingControlState) -> dict[str, Any]:
        super().apply(state)
        _bump_local(state, "u", self.patch_id, self.strength)
        _bump_local(state, "v", self.patch_id, self.strength)
        return self.metadata()


class IncreaseLocalPressureLoss(InterventionAction):
    name = "IncreaseLocalPressureLoss"

    def apply(self, state: TrainingControlState) -> dict[str, Any]:
        super().apply(state)
        _bump_local(state, "p", self.patch_id, self.strength)
        return self.metadata()


class IncreaseLocalPressureGradientLoss(InterventionAction):
    name = "IncreaseLocalPressureGradientLoss"

    def apply(self, state: TrainingControlState) -> dict[str, Any]:
        super().apply(state)
        _bump_local(state, "pressure_gradient", self.patch_id, self.strength)
        return self.metadata()


class AddLocalPressureAnchor(InterventionAction):
    name = "AddLocalPressureAnchor"

    def apply(self, state: TrainingControlState) -> dict[str, Any]:
        super().apply(state)
        state.pressure_anchor_patches[int(self.patch_id)] = self.strength
        state.active_aux_losses.add("pressure_anchor")
        return self.metadata()


class AddLocalPressurePoissonLoss(InterventionAction):
    name = "AddLocalPressurePoissonLoss"

    def apply(self, state: TrainingControlState) -> dict[str, Any]:
        super().apply(state)
        state.active_aux_losses.add("pressure_poisson")
        _bump_local(state, "pressure_poisson", self.patch_id, self.strength)
        return self.metadata()


class IncreaseLocalVorticityLoss(InterventionAction):
    name = "IncreaseLocalVorticityLoss"

    def apply(self, state: TrainingControlState) -> dict[str, Any]:
        super().apply(state)
        _bump_local(state, "omega", self.patch_id, self.strength)
        return self.metadata()


class AddLocalVorticityTransportLoss(InterventionAction):
    name = "AddLocalVorticityTransportLoss"

    def apply(self, state: TrainingControlState) -> dict[str, Any]:
        super().apply(state)
        state.active_aux_losses.add("vorticity_transport")
        _bump_local(state, "vorticity_transport", self.patch_id, self.strength)
        return self.metadata()


class IncreaseLocalDivergenceLoss(InterventionAction):
    name = "IncreaseLocalDivergenceLoss"

    def apply(self, state: TrainingControlState) -> dict[str, Any]:
        super().apply(state)
        _bump_local(state, "continuity", self.patch_id, self.strength)
        return self.metadata()


class IncreaseLocalMomentumLoss(InterventionAction):
    name = "IncreaseLocalMomentumLoss"

    def apply(self, state: TrainingControlState) -> dict[str, Any]:
        super().apply(state)
        _bump_local(state, "momentum_u", self.patch_id, self.strength)
        _bump_local(state, "momentum_v", self.patch_id, self.strength)
        return self.metadata()


class AddLocalBoundarySamples(InterventionAction):
    name = "AddLocalBoundarySamples"

    def apply(self, state: TrainingControlState) -> dict[str, Any]:
        super().apply(state)
        state.sampling_priorities[int(self.patch_id)] = state.sampling_priorities.get(int(self.patch_id), 0.0) + self.strength
        _bump_local(state, "bc", self.patch_id, self.strength)
        return self.metadata()


class IncreasePatchSamplingPriority(InterventionAction):
    name = "IncreasePatchSamplingPriority"

    def apply(self, state: TrainingControlState) -> dict[str, Any]:
        super().apply(state)
        state.sampling_priorities[int(self.patch_id)] = state.sampling_priorities.get(int(self.patch_id), 0.0) + self.strength
        return self.metadata()


class GenericResidualRefinement(InterventionAction):
    name = "GenericResidualRefinement"

    def apply(self, state: TrainingControlState) -> dict[str, Any]:
        super().apply(state)
        _bump_local(state, "pde", self.patch_id, self.strength)
        state.sampling_priorities[int(self.patch_id)] = state.sampling_priorities.get(int(self.patch_id), 0.0) + self.strength
        return self.metadata()


class NoOpAction(InterventionAction):
    name = "NoOpAction"
