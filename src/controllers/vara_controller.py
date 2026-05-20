"""Variable-aware regional adaptive controller."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .actions import TrainingControlState, cap_control_state
from .constrained_policy import ConstrainedVARAPolicy
from .rule_based_policy import RuleBasedVARAPolicy


class VARAController:
    """Separate diagnosis, policy, intervention, and acceptance bookkeeping."""

    def __init__(
        self,
        initial_weights: dict[str, float],
        policy: RuleBasedVARAPolicy | ConstrainedVARAPolicy | None = None,
        constrained: bool = False,
    ) -> None:
        self.state = TrainingControlState(global_weights=dict(initial_weights))
        self.policy = policy or (ConstrainedVARAPolicy() if constrained else RuleBasedVARAPolicy())
        self.constrained = constrained
        self.pending: dict[str, Any] | None = None
        self.rejected_counts: dict[tuple[str, int], int] = {}
        self.variable_history: list[dict[str, Any]] = []
        self.variable_strength_factors: dict[str, float] = {}

    def step(
        self,
        cycle: int,
        weak_regions: list[object],
        scores: np.ndarray,
        diagnostic_names: list[str],
        stage: str = "train",
    ) -> dict[str, Any]:
        """Apply policy actions and return a log record."""
        actions = self.policy.propose(weak_regions, self.state, stage)
        action_logs = []
        target_pairs: list[tuple[int, int]] = []
        name_to_idx = {name: i for i, name in enumerate(diagnostic_names)}
        for wr in weak_regions:
            if wr.variable in name_to_idx:
                target_pairs.append((name_to_idx[wr.variable], wr.patch_id))
        for action in actions:
            action_logs.append(action.apply(self.state))
        record = {
            "cycle": cycle,
            "stage": stage,
            "weak_regions": [asdict(wr) if hasattr(wr, "__dataclass_fields__") else dict(wr) for wr in weak_regions],
            "actions": action_logs,
            "global_weights": dict(self.state.global_weights),
            "local_weights": {k: dict(v) for k, v in self.state.local_weights.items()},
            "sampling_priorities": dict(self.state.sampling_priorities),
            "active_aux_losses": sorted(self.state.active_aux_losses),
        }
        self.pending = {
            "cycle": cycle,
            "before_scores": np.array(scores, copy=True),
            "target_pairs": target_pairs,
            "actions": actions,
        }
        return record

    def propose_trial(
        self,
        cycle: int,
        weak_regions: list[object],
        diagnostic_names: list[str],
        previous_decision: dict[str, Any] | None = None,
        stage: str = "trial",
    ) -> dict[str, Any]:
        """Apply one constrained trial intervention and keep a snapshot for rollback."""
        selected = weak_regions[0] if weak_regions else None
        targeted_variable = selected.variable if selected is not None else "none"
        targeted_patch = selected.patch_id if selected is not None else None
        strength_factor = self._strength_factor(targeted_variable, targeted_patch, previous_decision)
        state_snapshot = self.state.snapshot()
        actions = self.policy.propose([selected] if selected is not None else [], self.state, stage)
        for action in actions:
            action.strength *= strength_factor
        action_logs = [action.apply(self.state) for action in actions]
        cap_control_state(self.state)
        trial = {
            "cycle": cycle,
            "stage": stage,
            "targeted_variable": targeted_variable,
            "targeted_patch": targeted_patch,
            "targeted_metric": target_metric_for_variable(targeted_variable),
            "weak_region": asdict(selected) if selected is not None and hasattr(selected, "__dataclass_fields__") else None,
            "actions": action_logs,
            "state_snapshot": state_snapshot,
            "strength_factor": strength_factor,
            "old_weights": state_snapshot["global_weights"],
            "old_local_weights": state_snapshot["local_weights"],
            "old_sampling_priorities": state_snapshot["sampling_priorities"],
            "new_weights": dict(self.state.global_weights),
            "new_local_weights": {k: dict(v) for k, v in self.state.local_weights.items()},
            "new_sampling_priorities": dict(self.state.sampling_priorities),
        }
        return trial

    def rollback_trial(self, trial: dict[str, Any]) -> None:
        """Rollback controller intervention state after a rejected trial."""
        self.state.restore(trial["state_snapshot"])
        key = (str(trial["targeted_variable"]), int(trial["targeted_patch"] or -1))
        self.rejected_counts[key] = self.rejected_counts.get(key, 0) + 1
        self.variable_strength_factors[str(trial["targeted_variable"])] = (
            self.variable_strength_factors.get(str(trial["targeted_variable"]), 1.0) * 0.5
        )

    def record_decision(self, decision: dict[str, Any]) -> None:
        """Track variable-level improvement history for damping."""
        self.variable_history.append(decision)
        if len(self.variable_history) > 8:
            self.variable_history = self.variable_history[-8:]

    def apply_pressure_protection(self) -> dict[str, float]:
        """Bias the accepted controller state away from pressure collateral damage."""
        weights = self.state.global_weights
        weights["p"] = min(float(weights.get("p", 1.0)) * 1.05, 3.0)
        weights["pressure_gradient"] = min(float(weights.get("pressure_gradient", 0.2)) * 1.10, 3.0)
        for name in ("u", "v"):
            weights[name] = min(float(weights.get(name, 1.0)), 2.0)
            if name in self.state.local_weights:
                self.state.local_weights[name] = {
                    int(pid): 0.5 * float(value) for pid, value in self.state.local_weights[name].items()
                }
            self.variable_strength_factors[name] = self.variable_strength_factors.get(name, 1.0) * 0.5
        cap_control_state(self.state)
        return dict(weights)

    def _strength_factor(
        self,
        variable: str,
        patch_id: int | None,
        previous_decision: dict[str, Any] | None,
    ) -> float:
        factor = self.variable_strength_factors.get(str(variable), 1.0)
        key = (str(variable), int(patch_id or -1))
        if self.rejected_counts.get(key, 0) > 0:
            factor *= 0.5 ** self.rejected_counts[key]
        if previous_decision is not None:
            same_variable = previous_decision.get("targeted_variable") == variable
            poor_improvement = float(previous_decision.get("targeted_improvement", 0.0)) < float(
                getattr(self.policy, "min_improvement", 0.005)
            )
            repeated = int(previous_decision.get("repeat_count", 1)) >= 2
            if same_variable and poor_improvement and repeated:
                factor *= 0.5
        return max(0.0625, float(factor))

    def evaluate_pending(self, after_scores: np.ndarray) -> dict[str, Any] | None:
        """Evaluate constrained acceptance on the previous cycle."""
        if not self.constrained or self.pending is None or not isinstance(self.policy, ConstrainedVARAPolicy):
            return None
        accepted, metrics = self.policy.accept(
            self.pending["before_scores"],
            after_scores,
            self.pending["target_pairs"],
        )
        if not accepted:
            for action in reversed(self.pending["actions"]):
                action.rollback(self.state)
        record = {
            "cycle": self.pending["cycle"],
            "accepted": accepted,
            "rollback": not accepted,
            **metrics,
        }
        self.pending = None
        return record


def target_metric_for_variable(variable: str) -> str:
    """Map diagnostic variables/actions to validation metrics."""
    variable = str(variable)
    if "p_error" in variable or "pressure" in variable:
        return "p_rel_l2_centered"
    if "v_error" in variable:
        return "v_rel_l2"
    if "u_error" in variable:
        return "u_rel_l2"
    if "speed" in variable:
        return "u_rel_l2"
    if "omega" in variable or "vorticity" in variable:
        return "omega_rel_l2"
    if "residual" in variable or "momentum" in variable or "continuity" in variable or "boundary" in variable:
        return "pde_residual_mean"
    return "u_rel_l2"
