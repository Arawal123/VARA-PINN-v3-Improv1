"""Mode specifications for baselines and clean VARA ablations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


BASELINE_MODE_SPECS: dict[str, dict[str, Any]] = {
    "vanilla_pinn": {
        "algorithm_family": "uniform_fixed_weight_pinn",
        "adaptive_sampling": False,
        "fixed_loss_weights": True,
    },
    "residual_adaptive_sampling": {
        "algorithm_family": "rar_rad_residual_sampling",
        "adaptive_sampling": True,
        "fixed_loss_weights": True,
    },
    "global_adaptive_loss": {
        "algorithm_family": "global_dynamic_loss_balancing",
        "adaptive_sampling": False,
        "global_adaptive_loss": True,
    },
    "self_adaptive_attention": {
        "algorithm_family": "batchwise_self_adaptive_attention",
        "adaptive_sampling": False,
        "self_adaptive_attention": True,
    },
    "gpinn": {
        "algorithm_family": "residual_gradient_penalty_pinn",
        "adaptive_sampling": False,
        "gpinn_gradient_penalty": True,
        "gpinn_weight": 0.1,
    },
    "hard_divergence_pinn": {
        "algorithm_family": "streamfunction_hard_divergence_pinn",
        "adaptive_sampling": False,
        "model_kind": "streamfunction_p",
    },
    "more_points_vanilla": {
        "algorithm_family": "uniform_fixed_weight_more_points",
        "adaptive_sampling": False,
        "point_budget_multiplier": 2.0,
    },
}


ABLATION_MODE_SPECS: dict[str, dict[str, Any]] = {
    "s_only": {"components": {"V": False, "R": False, "S": True, "A": False, "B": False}},
    "v_only": {"components": {"V": True, "R": False, "S": False, "A": False, "B": False}},
    "r_s": {"components": {"V": False, "R": True, "S": True, "A": False, "B": False}},
    "v_r": {"components": {"V": True, "R": True, "S": False, "A": False, "B": False}},
    "v_r_s": {"components": {"V": True, "R": True, "S": True, "A": False, "B": False}},
    "v_r_s_no_a": {"components": {"V": True, "R": True, "S": True, "A": False, "B": True}},
    "full_vara_factorial": {"components": {"V": True, "R": True, "S": True, "A": True, "B": True}},
    "full_vara_streamfunction": {
        "components": {"V": True, "R": True, "S": True, "A": True, "B": True},
        "model_kind": "streamfunction_p",
    },
    "full_vara_direct": {
        "components": {"V": True, "R": True, "S": True, "A": True, "B": True},
        "model_kind": "direct_uvp",
    },
}


def prepare_mode_config(config: dict[str, Any], mode: str) -> dict[str, Any]:
    """Return a config with explicit algorithm metadata and mode-required overrides."""
    cfg = deepcopy(config)
    cfg["method"] = cfg.get("method", mode)
    if mode in BASELINE_MODE_SPECS:
        spec = deepcopy(BASELINE_MODE_SPECS[mode])
        cfg["baseline"] = {**cfg.get("baseline", {}), **spec, "mode": mode}
        if "model_kind" in spec:
            cfg["model"] = {**cfg.get("model", {}), "kind": spec["model_kind"]}
        if spec.get("gpinn_gradient_penalty"):
            weights = dict(cfg.get("training", {}).get("weights", {}))
            weights["gpinn"] = float(cfg.get("baseline", {}).get("gpinn_weight", spec.get("gpinn_weight", 0.1)))
            cfg["training"] = {**cfg.get("training", {}), "weights": weights}
        multiplier = float(spec.get("point_budget_multiplier", 1.0))
        if multiplier != 1.0:
            training = dict(cfg.get("training", {}))
            training["n_collocation"] = int(round(int(training.get("n_collocation", 1024)) * multiplier))
            training["compute_budget_note"] = f"n_collocation multiplied by {multiplier:g} for more-points baseline"
            cfg["training"] = training
        return cfg
    if mode in ABLATION_MODE_SPECS:
        spec = deepcopy(ABLATION_MODE_SPECS[mode])
        cfg["ablation"] = {**cfg.get("ablation", {}), **spec, "mode": mode}
        if "model_kind" in spec:
            cfg["model"] = {**cfg.get("model", {}), "kind": spec["model_kind"]}
    return cfg

