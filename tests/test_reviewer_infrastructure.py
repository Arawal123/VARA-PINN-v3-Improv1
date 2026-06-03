from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_benchmark_results import build_methodwise, build_paired_statistics_table
from src.training.baseline_modes import BASELINE_MODE_SPECS, prepare_mode_config
from src.training.baseline_trainer import BaselineTrainer


def test_baseline_modes_have_distinct_algorithm_specs():
    families = {mode: spec["algorithm_family"] for mode, spec in BASELINE_MODE_SPECS.items()}
    assert families["vanilla_pinn"] != families["residual_adaptive_sampling"]
    assert BASELINE_MODE_SPECS["residual_adaptive_sampling"]["adaptive_sampling"] is True
    assert BASELINE_MODE_SPECS["gpinn"]["gpinn_gradient_penalty"] is True
    assert BASELINE_MODE_SPECS["hard_divergence_pinn"]["model_kind"] == "streamfunction_p"


def test_prepare_mode_config_changes_hard_divergence_and_more_points():
    base = {"model": {"hidden_layers": [8]}, "training": {"n_collocation": 10, "weights": {"pde": 1.0}}}
    hard = prepare_mode_config(base, "hard_divergence_pinn")
    more = prepare_mode_config(base, "more_points_vanilla")
    assert hard["model"]["kind"] == "streamfunction_p"
    assert more["training"]["n_collocation"] == 20


def test_matched_compute_metadata_logged_for_tiny_baseline(tmp_path):
    config = {
        "benchmark": "lid_driven_cavity",
        "seed": 0,
        "deterministic": True,
        "device": "cpu",
        "run_type": "smoke",
        "benchmark_params": {"reynolds": 100.0, "reference": "ghia", "profile_only": True},
        "model": {"input_dim": 2, "output_dim": 3, "hidden_layers": [8], "activation": "tanh"},
        "optimizer": {"lr": 1e-3},
        "training": {
            "adaptive_cycles": 1,
            "epochs_per_cycle": 1,
            "n_collocation": 8,
            "n_boundary": 8,
            "n_data": 0,
            "weights": {"pde": 1.0, "momentum_u": 1.0, "momentum_v": 1.0, "continuity": 1.0, "bc": 1.0},
        },
        "validation": {"nx": 5, "ny": 5},
        "test": {"nx": 5, "ny": 5},
        "diagnostics": {"mode": "residual_only", "variables": ["aggregate_pde_residual"]},
        "patches": {"nx_patches": 2, "ny_patches": 2},
        "weak_regions": {"top_k_per_variable": 1},
        "experiments": {"root": str(tmp_path)},
    }
    trainer = BaselineTrainer(config, mode="vanilla_pinn")
    metrics = trainer.run()
    assert metrics["optimizer_steps"] == 1
    assert metrics["collocation_points_seen"] == 8
    assert metrics["boundary_points_seen"] == 8
    assert metrics["parameter_count"] > 0


def test_aggregation_reports_valid_n_and_paired_stats():
    df = pd.DataFrame(
        [
            {"benchmark": "b", "run_type": "paper", "method": "vanilla", "seed": 0, "u_rmse": 2.0},
            {"benchmark": "b", "run_type": "paper", "method": "vara", "seed": 0, "u_rmse": 1.0},
            {"benchmark": "b", "run_type": "paper", "method": "vanilla", "seed": 1, "u_rmse": 4.0},
            {"benchmark": "b", "run_type": "paper", "method": "vara", "seed": 1, "u_rmse": 2.0},
        ]
    )
    methodwise = build_methodwise(df)
    assert int(methodwise.loc[methodwise["method"] == "vanilla", "u_rmse_valid_n"].iloc[0]) == 2
    paired = build_paired_statistics_table(df)
    row = paired[(paired["candidate_method"] == "vara") & (paired["metric"] == "u_rmse")].iloc[0]
    assert int(row["valid_n"]) == 2
    assert row["paired_improvement_percent_mean"] == 50.0
