import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.controllers.local_controller import LocalControllerConfig, LocalIntervention, LocalVARAController
from src.training.trainer import ExperimentTrainer


def test_smoke_runs_are_not_collapse_evaluated():
    trainer = object.__new__(ExperimentTrainer)
    trainer.config = {"collapse_thresholds": {}}
    metrics = {
        "run_type": "smoke",
        "reportable": False,
        "collapse_evaluated": False,
        "has_reference": True,
        "u_rel_l2": 100.0,
        "unweighted_validation_loss": 100.0,
        "final_total_loss": 100.0,
    }
    assert trainer._collapsed(metrics) is False


def test_weighted_final_loss_alone_does_not_mark_collapse():
    trainer = object.__new__(ExperimentTrainer)
    trainer.config = {"collapse_thresholds": {}}
    metrics = {
        "collapse_evaluated": True,
        "has_reference": True,
        "u_rel_l2": 0.1,
        "v_rel_l2": math.nan,
        "p_rel_l2_centered": 0.1,
        "omega_rel_l2": 0.1,
        "u_rmse": 0.1,
        "v_rmse": 0.1,
        "p_rmse_centered": 0.1,
        "omega_rmse": 0.1,
        "pde_residual_mean": 0.1,
        "continuity_residual_mean": 0.1,
        "momentum_residual_mean": 0.1,
        "unweighted_validation_loss": 0.1,
        "boundary_condition_error": 0.1,
        "final_total_loss": 1.0e6,
    }
    assert trainer._collapsed(metrics) is False


def test_local_objective_uses_rmse_fallback_for_zero_reference_fields():
    controller = LocalVARAController(
        initial_weights={},
        config=LocalControllerConfig.from_dict({"objective_weights": {"v": 1.0}}),
    )
    assert controller.objective({"v_rel_l2": math.nan, "v_rmse": 0.25}) == 0.25


def test_local_acceptance_rejects_continuity_collateral_damage():
    controller = LocalVARAController(
        initial_weights={},
        config=LocalControllerConfig.from_dict(
            {
                "min_improvement": 0.01,
                "continuity_collateral_tolerance": 0.05,
                "objective_weights": {"u": 1.0, "continuity": 1.0},
            }
        ),
    )
    intervention = LocalIntervention(
        variable="u_error",
        patch_id=0,
        action="increase_local_velocity",
        loss_variables=["u"],
        strength=1.0,
        severity=1.0,
        confidence=1.0,
        bounds=(0.0, 1.0, 0.0, 1.0, None, None),
    )
    before_scores = np.array([[1.0]])
    after_scores = np.array([[0.8]])
    before_metrics = {
        "u_rel_l2": 1.0,
        "u_rmse": 1.0,
        "continuity_residual_mean": 1.0,
        "unweighted_validation_loss": 1.0,
    }
    after_metrics = {
        "u_rel_l2": 0.8,
        "u_rmse": 0.8,
        "continuity_residual_mean": 1.2,
        "unweighted_validation_loss": 1.0,
    }
    accepted, decision = controller.evaluate_acceptance(
        [intervention],
        before_scores,
        after_scores,
        ["u_error"],
        before_metrics,
        after_metrics,
        constrained=True,
    )
    assert accepted is False
    assert decision["continuity_collateral_damage"] > 0.05
