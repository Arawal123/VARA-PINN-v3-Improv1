"""Local variable-region controller for VARA-PINN."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .actions import TrainingControlState


METRIC_BY_DIAGNOSTIC = {
    "u_error": "u_rel_l2",
    "v_error": "v_rel_l2",
    "p_error_mean_centered": "p_rel_l2_centered",
    "pressure_gradient_error": "p_rel_l2_centered",
    "omega_error": "omega_rel_l2",
    "continuity_residual": "continuity_residual_mean",
    "momentum_u_residual": "momentum_residual_mean",
    "momentum_v_residual": "momentum_residual_mean",
    "aggregate_pde_residual": "pde_residual_mean",
    "pde_residual": "pde_residual_mean",
    "boundary_violation": "boundary_condition_error",
    "u_profile_error": "u_centerline_rmse",
    "v_profile_error": "v_centerline_rmse",
    "profile_error": "centerline_profile_score",
}


@dataclass
class LocalControllerConfig:
    """Configuration for local variable-region control."""

    min_improvement: float = 0.005
    collateral_tolerance: float = 0.10
    pressure_collateral_tolerance: float = 0.05
    trial_epochs: int = 100
    max_actions_per_cycle: int = 6
    initial_strength: float = 1.0
    damping_factor: float = 0.5
    local_velocity_weight_max: float = 2.0
    local_pressure_weight_max: float = 3.0
    local_omega_weight_max: float = 2.0
    local_pde_weight_max: float = 2.0
    local_bc_weight_max: float = 7.0
    objective_weights: dict[str, float] | None = None
    tiny_j_tolerance: float = 0.002
    strong_target_factor: float = 2.0
    continuity_collateral_tolerance: float = 0.05
    boundary_collateral_tolerance: float = 0.05
    validation_loss_tolerance: float = 0.02

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LocalControllerConfig":
        data = dict(data or {})
        return cls(
            min_improvement=float(data.get("min_improvement", 0.005)),
            collateral_tolerance=float(data.get("collateral_tolerance", 0.10)),
            pressure_collateral_tolerance=float(data.get("pressure_collateral_tolerance", 0.05)),
            trial_epochs=int(data.get("trial_epochs", 100)),
            max_actions_per_cycle=int(data.get("max_actions_per_cycle", 6)),
            initial_strength=float(data.get("initial_strength", 1.0)),
            damping_factor=float(data.get("damping_factor", 0.5)),
            local_velocity_weight_max=float(data.get("local_velocity_weight_max", 2.0)),
            local_pressure_weight_max=float(data.get("local_pressure_weight_max", 3.0)),
            local_omega_weight_max=float(data.get("local_omega_weight_max", 2.0)),
            local_pde_weight_max=float(data.get("local_pde_weight_max", 2.0)),
            local_bc_weight_max=float(data.get("local_bc_weight_max", 7.0)),
            objective_weights=dict(data.get("objective_weights", {"u": 1.0, "v": 1.0, "p": 1.5, "omega": 1.0, "residual": 0.25})),
            tiny_j_tolerance=float(data.get("tiny_j_tolerance", 0.002)),
            strong_target_factor=float(data.get("strong_target_factor", 2.0)),
            continuity_collateral_tolerance=float(data.get("continuity_collateral_tolerance", 0.05)),
            boundary_collateral_tolerance=float(data.get("boundary_collateral_tolerance", 0.05)),
            validation_loss_tolerance=float(data.get("validation_loss_tolerance", 0.02)),
        )

    @property
    def caps(self) -> dict[str, float]:
        return {
            "u": self.local_velocity_weight_max,
            "v": self.local_velocity_weight_max,
            "p": self.local_pressure_weight_max,
            "pressure_gradient": self.local_pressure_weight_max,
            "omega": self.local_omega_weight_max,
            "pde": self.local_pde_weight_max,
            "momentum_u": self.local_pde_weight_max,
            "momentum_v": self.local_pde_weight_max,
            "continuity": self.local_pde_weight_max,
            "bc": self.local_bc_weight_max,
            "u_profile": self.local_velocity_weight_max,
            "v_profile": self.local_velocity_weight_max,
            "profile": self.local_velocity_weight_max,
        }


@dataclass
class LocalIntervention:
    """One variable-region local intervention."""

    variable: str
    patch_id: int
    action: str
    loss_variables: list[str]
    strength: float
    severity: float
    confidence: float
    bounds: tuple[float, float, float, float, float | None, float | None]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LocalPairState:
    """Persistent controller memory for one variable-patch pair."""

    strength: float
    accepted_count: int = 0
    rejected_count: int = 0
    last_target_improvement: float = 0.0
    last_collateral: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class LocalVARAController:
    """Controller that only modifies variable-region local weights and priorities."""

    def __init__(self, initial_weights: dict[str, float], config: LocalControllerConfig) -> None:
        self.state = TrainingControlState(global_weights=dict(initial_weights))
        self.config = config
        self.pair_state: dict[tuple[str, int], LocalPairState] = {}
        self.history: list[dict[str, Any]] = []

    def snapshot(self) -> dict[str, Any]:
        return self.state.snapshot()

    def rollback(self, snapshot: dict[str, Any]) -> None:
        self.state.restore(snapshot)

    def propose(self, weak_regions: list[object]) -> list[LocalIntervention]:
        interventions: list[LocalIntervention] = []
        seen: set[tuple[str, int, str]] = set()
        for region in weak_regions:
            action, loss_variables = self._action_for_variable(region.variable)
            key = (region.variable, int(region.patch_id), action)
            if key in seen:
                continue
            seen.add(key)
            strength = self._strength(region.variable, int(region.patch_id), float(region.confidence))
            interventions.append(
                LocalIntervention(
                    variable=region.variable,
                    patch_id=int(region.patch_id),
                    action=action,
                    loss_variables=loss_variables,
                    strength=strength,
                    severity=float(region.severity),
                    confidence=float(region.confidence),
                    bounds=region.bounds,
                )
            )
            if len(interventions) >= self.config.max_actions_per_cycle:
                break
        return interventions

    def apply(self, interventions: list[LocalIntervention]) -> None:
        for intervention in interventions:
            for loss_variable in intervention.loss_variables:
                self._bump_local(loss_variable, intervention.patch_id, intervention.strength)
            self.state.sampling_priorities[intervention.patch_id] = (
                self.state.sampling_priorities.get(intervention.patch_id, 0.0) + intervention.strength
            )

    def objective(self, metrics: dict[str, float]) -> float:
        weights = self.config.objective_weights or {}
        return float(
            float(weights.get("u", 1.0)) * self._metric_with_fallback(metrics, "u_rel_l2", "u_rmse")
            + float(weights.get("v", 1.0)) * self._metric_with_fallback(metrics, "v_rel_l2", "v_rmse")
            + float(weights.get("p", 1.5)) * self._metric_with_fallback(metrics, "p_rel_l2_centered", "p_rmse_centered")
            + float(weights.get("omega", 1.0)) * self._metric_with_fallback(metrics, "omega_rel_l2", "omega_rmse")
            + float(weights.get("residual", 0.25)) * self._metric(metrics, "pde_residual_mean")
            + float(weights.get("continuity", 0.0)) * self._metric(metrics, "continuity_residual_mean")
            + float(weights.get("momentum", 0.0)) * self._metric(metrics, "momentum_residual_mean")
            + float(weights.get("boundary", 0.0)) * self._metric(metrics, "boundary_condition_error")
            + float(weights.get("unweighted_validation", 0.0)) * self._metric(metrics, "unweighted_validation_loss")
            + float(weights.get("profile", 0.0)) * self._metric(metrics, "centerline_profile_score")
            + float(weights.get("cavity", 0.0)) * self._metric(metrics, "cavity_benchmark_score")
        )

    def evaluate_acceptance(
        self,
        interventions: list[LocalIntervention],
        before_raw_scores: np.ndarray,
        after_raw_scores: np.ndarray,
        diagnostic_names: list[str],
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
        constrained: bool,
    ) -> tuple[bool, dict[str, Any]]:
        if not constrained:
            target_improvement = self._target_improvement(interventions, before_raw_scores, after_raw_scores, diagnostic_names)
            return True, self._decision_metrics(target_improvement, before_metrics, after_metrics, interventions)

        target_improvement = self._target_improvement(interventions, before_raw_scores, after_raw_scores, diagnostic_names)
        metrics = self._decision_metrics(target_improvement, before_metrics, after_metrics, interventions)
        j_ok = metrics["J_after"] <= metrics["J_before"]
        tiny_j_ok = (
            metrics["J_after"] <= metrics["J_before"] * (1.0 + self.config.tiny_j_tolerance)
            and target_improvement >= self.config.min_improvement * self.config.strong_target_factor
        )
        accepted = (
            target_improvement >= self.config.min_improvement
            and (j_ok or tiny_j_ok)
            and metrics["max_collateral_damage"] <= self.config.collateral_tolerance
            and metrics["pressure_collateral_damage"] <= self.config.pressure_collateral_tolerance
            and metrics["continuity_collateral_damage"] <= self.config.continuity_collateral_tolerance
            and metrics["boundary_collateral_damage"] <= self.config.boundary_collateral_tolerance
            and metrics["validation_loss_damage"] <= self.config.validation_loss_tolerance
        )
        return accepted, metrics

    def record_decision(self, interventions: list[LocalIntervention], decision: dict[str, Any]) -> None:
        self.history.append(decision)

    def mark_accepted(self, intervention: LocalIntervention, decision: dict[str, Any]) -> None:
        """Update persistent pair memory after an accepted action."""
        state = self._pair_state(intervention.variable, intervention.patch_id)
        state.accepted_count += 1
        state.last_target_improvement = float(decision.get("target_local_improvement", 0.0))
        state.last_collateral = float(decision.get("max_collateral_damage", 0.0))
        if state.last_target_improvement < self.config.min_improvement:
            state.strength = max(1e-4, state.strength * self.config.damping_factor)

    def mark_rejected(self, intervention: LocalIntervention, decision: dict[str, Any]) -> None:
        """Update persistent pair memory after a rejected action."""
        state = self._pair_state(intervention.variable, intervention.patch_id)
        state.rejected_count += 1
        state.last_target_improvement = float(decision.get("target_local_improvement", 0.0))
        state.last_collateral = float(decision.get("max_collateral_damage", 0.0))
        state.strength = max(1e-4, state.strength * self.config.damping_factor)

    def pair_state_record(self) -> dict[str, Any]:
        """Serialize persistent pair controller state for logs."""
        return {f"{variable}|{patch_id}": state.to_record() for (variable, patch_id), state in self.pair_state.items()}

    def active_patches(self) -> set[int]:
        patches = set(int(pid) for pid in self.state.sampling_priorities)
        for weights in self.state.local_weights.values():
            patches.update(int(pid) for pid in weights)
        return patches

    def _bump_local(self, variable: str, patch_id: int, amount: float) -> None:
        cap = self.config.caps.get(variable, self.config.local_pde_weight_max)
        self.state.local_weights.setdefault(variable, {})
        old = float(self.state.local_weights[variable].get(int(patch_id), 0.0))
        self.state.local_weights[variable][int(patch_id)] = min(old + float(amount), cap)

    def _strength(self, variable: str, patch_id: int, confidence: float) -> float:
        state = self._pair_state(variable, patch_id)
        strength = state.strength * max(0.5, confidence)
        attempted = state.accepted_count + state.rejected_count
        if attempted > 0 and state.last_target_improvement < self.config.min_improvement:
            strength *= self.config.damping_factor
        return max(1e-4, float(strength))

    def _pair_state(self, variable: str, patch_id: int) -> LocalPairState:
        key = (variable, int(patch_id))
        if key not in self.pair_state:
            self.pair_state[key] = LocalPairState(strength=float(self.config.initial_strength))
        return self.pair_state[key]

    def _action_for_variable(self, variable: str) -> tuple[str, list[str]]:
        if "u_error" in variable:
            return "increase_local_velocity", ["u", "momentum_u"]
        if "v_error" in variable:
            return "increase_local_velocity", ["v", "momentum_v"]
        if "p_error" in variable or "pressure" in variable:
            return "increase_local_pressure", ["p", "pressure_gradient"]
        if "omega" in variable or "vorticity" in variable:
            return "increase_local_vorticity", ["omega", "pde"]
        if "continuity" in variable:
            return "increase_local_divergence", ["continuity", "pde"]
        if "momentum_u" in variable:
            return "increase_local_momentum", ["momentum_u", "pde"]
        if "momentum_v" in variable:
            return "increase_local_momentum", ["momentum_v", "pde"]
        if "u_profile" in variable:
            return "increase_local_cavity_profile", ["u_profile"]
        if "v_profile" in variable:
            return "increase_local_cavity_profile", ["v_profile"]
        if "profile" in variable:
            return "increase_local_cavity_profile", ["profile"]
        if "boundary" in variable:
            return "increase_local_boundary", ["bc"]
        return "increase_local_pde", ["pde"]

    def _target_improvement(
        self,
        interventions: list[LocalIntervention],
        before_raw_scores: np.ndarray,
        after_raw_scores: np.ndarray,
        diagnostic_names: list[str],
    ) -> float:
        eps = 1e-12
        name_to_idx = {name: i for i, name in enumerate(diagnostic_names)}
        values = []
        for intervention in interventions:
            if intervention.variable not in name_to_idx:
                continue
            j = name_to_idx[intervention.variable]
            r = intervention.patch_id
            before = float(before_raw_scores[j, r])
            after = float(after_raw_scores[j, r])
            values.append((before - after) / (abs(before) + eps))
        return float(np.mean(values)) if values else 0.0

    def _decision_metrics(
        self,
        target_improvement: float,
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
        interventions: list[LocalIntervention],
    ) -> dict[str, Any]:
        metric_targets = {METRIC_BY_DIAGNOSTIC.get(intervention.variable, "pde_residual_mean") for intervention in interventions}
        collateral = {}
        for metric_name in [
            "u_rel_l2",
            "v_rel_l2",
            "p_rel_l2_centered",
            "omega_rel_l2",
            "pde_residual_mean",
            "continuity_residual_mean",
            "momentum_residual_mean",
            "boundary_condition_error",
            "unweighted_validation_loss",
            "u_centerline_rmse",
            "v_centerline_rmse",
            "centerline_profile_score",
            "cavity_benchmark_score",
        ]:
            before = self._metric(before_metrics, metric_name)
            after = self._metric(after_metrics, metric_name)
            if metric_name in metric_targets:
                continue
            collateral[metric_name] = max(0.0, (after - before) / (abs(before) + 1e-12))
        pressure_before = self._metric(before_metrics, "p_rel_l2_centered")
        pressure_after = self._metric(after_metrics, "p_rel_l2_centered")
        return {
            "target_local_improvement": float(target_improvement),
            "J_before": self.objective(before_metrics),
            "J_after": self.objective(after_metrics),
            "max_collateral_damage": float(max(collateral.values()) if collateral else 0.0),
            "pressure_collateral_damage": float(max(0.0, (pressure_after - pressure_before) / (abs(pressure_before) + 1e-12))),
            "continuity_collateral_damage": collateral.get("continuity_residual_mean", 0.0),
            "boundary_collateral_damage": collateral.get("boundary_condition_error", 0.0),
            "validation_loss_damage": collateral.get("unweighted_validation_loss", 0.0),
        }

    def _metric(self, metrics: dict[str, float], name: str) -> float:
        try:
            value = float(metrics.get(name, 0.0))
        except (TypeError, ValueError):
            return 0.0
        return value if np.isfinite(value) else 0.0

    def _metric_with_fallback(self, metrics: dict[str, float], primary: str, fallback: str) -> float:
        primary_value = self._metric(metrics, primary)
        if primary_value != 0.0:
            return primary_value
        return self._metric(metrics, fallback)
