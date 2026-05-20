"""Deterministic interpretable VARA policy."""

from __future__ import annotations

from .actions import (
    AddLocalBoundarySamples,
    AddLocalPressureAnchor,
    GenericResidualRefinement,
    IncreaseLocalDivergenceLoss,
    IncreaseLocalMomentumLoss,
    IncreaseLocalPressureGradientLoss,
    IncreaseLocalPressureLoss,
    IncreaseLocalVelocityLoss,
    IncreaseLocalVorticityLoss,
    IncreasePatchSamplingPriority,
    InterventionAction,
)


class RuleBasedVARAPolicy:
    """Map weak variable-region pairs to local intervention actions."""

    def __init__(self, strength: float = 1.0, max_actions_per_cycle: int = 8) -> None:
        self.strength = float(strength)
        self.max_actions_per_cycle = int(max_actions_per_cycle)

    def propose(self, weak_regions: list[object], state: object | None = None, stage: str = "train") -> list[InterventionAction]:
        actions: list[InterventionAction] = []
        for region in weak_regions[: self.max_actions_per_cycle]:
            actions.extend(self._actions_for_region(region))
            if len(actions) >= self.max_actions_per_cycle:
                break
        return actions[: self.max_actions_per_cycle]

    def _actions_for_region(self, region: object) -> list[InterventionAction]:
        name = region.variable
        patch_id = region.patch_id
        strength = self.strength * max(0.5, float(region.confidence))
        if "p_error" in name or name == "pressure" or "pressure_gradient" in name:
            return [
                IncreaseLocalPressureLoss(patch_id, name, strength),
                IncreaseLocalPressureGradientLoss(patch_id, name, 0.5 * strength),
                AddLocalPressureAnchor(patch_id, name, 0.25 * strength),
                IncreasePatchSamplingPriority(patch_id, name, strength),
            ]
        if "omega" in name or "vorticity" in name:
            return [
                IncreaseLocalVorticityLoss(patch_id, name, strength),
                IncreasePatchSamplingPriority(patch_id, name, strength),
            ]
        if "u_error" in name or "v_error" in name or "speed" in name:
            return [
                IncreaseLocalVelocityLoss(patch_id, name, strength),
                IncreasePatchSamplingPriority(patch_id, name, strength),
            ]
        if "continuity" in name:
            return [
                IncreaseLocalDivergenceLoss(patch_id, name, strength),
                IncreasePatchSamplingPriority(patch_id, name, 0.5 * strength),
            ]
        if "momentum" in name or "pde" in name or "residual" in name:
            return [
                IncreaseLocalMomentumLoss(patch_id, name, strength),
                GenericResidualRefinement(patch_id, name, strength),
            ]
        if "boundary" in name:
            return [AddLocalBoundarySamples(patch_id, name, strength)]
        return [GenericResidualRefinement(patch_id, name, strength)]

