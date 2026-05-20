"""Acceptance and rollback logic for controller interventions."""

from __future__ import annotations

import numpy as np

from .rule_based_policy import RuleBasedVARAPolicy


DEFAULT_OBJECTIVE_WEIGHTS = {
    "u_rel_l2": 1.0,
    "v_rel_l2": 1.0,
    "p_rel_l2_centered": 1.5,
    "omega_rel_l2": 1.0,
    "pde_residual_mean": 0.25,
}


class ConstrainedVARAPolicy:
    """Rule-based proposals with acceptance criteria."""

    def __init__(
        self,
        base_policy: RuleBasedVARAPolicy | None = None,
        min_improvement: float = 0.005,
        collateral_tolerance: float = 0.10,
        pressure_collateral_tolerance: float = 0.05,
        alpha_compute: float = 0.0,
        objective_weights: dict[str, float] | None = None,
    ) -> None:
        self.base_policy = base_policy or RuleBasedVARAPolicy()
        self.min_improvement = float(min_improvement)
        self.collateral_tolerance = float(collateral_tolerance)
        self.pressure_collateral_tolerance = float(pressure_collateral_tolerance)
        self.alpha_compute = float(alpha_compute)
        self.objective_weights = dict(objective_weights or DEFAULT_OBJECTIVE_WEIGHTS)

    def propose(self, weak_regions: list[object], state: object | None = None, stage: str = "train") -> list[object]:
        return self.base_policy.propose(weak_regions, state, stage)

    def accept(
        self,
        before_scores: np.ndarray,
        after_scores: np.ndarray,
        target_pairs: list[tuple[int, int]],
        compute_cost_delta: float = 0.0,
    ) -> tuple[bool, dict[str, float]]:
        """Accept if targeted improvement beats collateral damage."""
        eps = 1e-12
        improvements = []
        for j, r in target_pairs:
            b = before_scores[j, r]
            a = after_scores[j, r]
            improvements.append((b - a) / (b + eps))
        targeted = float(np.mean(improvements)) if improvements else 0.0
        degradation = np.maximum(after_scores - before_scores, 0.0)
        for j, r in target_pairs:
            degradation[j, r] = 0.0
        collateral = float(np.max(degradation)) if degradation.size else 0.0
        j_before = float(np.nanmean(before_scores))
        j_after = float(np.nanmean(after_scores) + self.alpha_compute * compute_cost_delta)
        accepted = j_after < j_before and collateral <= self.collateral_tolerance and targeted >= self.min_improvement
        return accepted, {
            "targeted_field_region_improvement": targeted,
            "spillover_degradation": collateral,
            "J_before": j_before,
            "J_after": j_after,
        }

    def score_metrics(self, metrics: dict[str, float]) -> float:
        """Compute the constrained controller objective J from validation metrics."""
        return float(sum(float(alpha) * float(metrics.get(name, 0.0)) for name, alpha in self.objective_weights.items()))

    def accept_metrics(
        self,
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
        targeted_metric: str,
        strong_j_margin: float = 0.10,
    ) -> tuple[bool, dict[str, float]]:
        """Accept a trial if it improves J, improves the target, and limits collateral damage."""
        eps = 1e-12
        j_before = self.score_metrics(before_metrics)
        j_after = self.score_metrics(after_metrics)
        before_target = float(before_metrics.get(targeted_metric, 0.0))
        after_target = float(after_metrics.get(targeted_metric, 0.0))
        targeted_improvement = (before_target - after_target) / (abs(before_target) + eps)

        collateral: dict[str, float] = {}
        for metric_name in self.objective_weights:
            if metric_name == targeted_metric:
                continue
            before = float(before_metrics.get(metric_name, 0.0))
            after = float(after_metrics.get(metric_name, 0.0))
            collateral[metric_name] = max(0.0, (after - before) / (abs(before) + eps))

        max_collateral = max(collateral.values()) if collateral else 0.0
        pressure_collateral = collateral.get("p_rel_l2_centered", 0.0)
        strong_j_improvement = j_after < j_before * (1.0 - strong_j_margin)
        pressure_ok = pressure_collateral <= self.pressure_collateral_tolerance or strong_j_improvement
        accepted = (
            j_after < j_before
            and targeted_improvement >= self.min_improvement
            and max_collateral <= self.collateral_tolerance
            and pressure_ok
        )
        return accepted, {
            "J_before": j_before,
            "J_after": j_after,
            "targeted_improvement": float(targeted_improvement),
            "pressure_collateral_damage": float(pressure_collateral),
            "max_collateral_damage": float(max_collateral),
            "pressure_protected": not pressure_ok,
        }
