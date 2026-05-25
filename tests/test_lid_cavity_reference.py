from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_on_grid
from src.diagnostics.diagnostic_maps import DiagnosticMapBuilder
from src.losses.base_losses import compute_pointwise_losses
from src.physics.cavity_reference import load_lid_cavity_profile_reference
from src.physics.rectangular_benchmarks import LidDrivenCavityQualitative


def test_builtin_ghia_profiles_load_for_required_reynolds():
    for reynolds in [100, 400, 1000]:
        ref = load_lid_cavity_profile_reference("ghia", reynolds)
        assert ref.has_u
        assert ref.has_v
        assert len(ref.u_profile) == 17
        assert len(ref.v_profile) == 17


def test_missing_ghia_reynolds_raises_clear_error():
    with pytest.raises(ValueError, match="Available Re"):
        load_lid_cavity_profile_reference("ghia", 250)


def test_cavity_profile_points_are_centerlines():
    bench = LidDrivenCavityQualitative(reynolds=100, reference="ghia")
    profile = bench.profile_reference_np()
    assert np.allclose(profile["u_xy"][:, 0], 0.5)
    assert np.allclose(profile["v_xy"][:, 1], 0.5)
    assert profile["u_ref"].shape == (17, 1)
    assert profile["v_ref"].shape == (17, 1)


def test_external_centerline_csv_loader(tmp_path):
    path = tmp_path / "cavity_profiles.csv"
    pd.DataFrame(
        [
            {"re": 100, "x": 0.5, "y": 0.0, "u_ref": 0.0, "v_ref": np.nan},
            {"re": 100, "x": 0.5, "y": 0.5, "u_ref": -0.2, "v_ref": np.nan},
            {"re": 100, "x": 0.0, "y": 0.5, "u_ref": np.nan, "v_ref": 0.0},
            {"re": 100, "x": 0.5, "y": 0.5, "u_ref": np.nan, "v_ref": 0.05},
        ]
    ).to_csv(path, index=False)
    ref = load_lid_cavity_profile_reference("external", 100, path)
    assert len(ref.u_profile) == 2
    assert len(ref.v_profile) == 2


def test_cavity_profile_metrics_are_finite():
    model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
    bench = LidDrivenCavityQualitative(reynolds=100, reference="ghia")
    _, _, coords = bench.grid(8, 8)
    metrics = evaluate_on_grid(model, bench, coords, torch.device("cpu"), steady=True)
    assert np.isfinite(metrics["u_centerline_rmse"])
    assert np.isfinite(metrics["v_centerline_rmse"])
    assert np.isfinite(metrics["centerline_profile_score"])
    assert np.isfinite(metrics["cavity_benchmark_score"])
    assert np.isnan(metrics["u_rel_l2"])


def test_cavity_profile_losses_are_added_to_training_batch():
    model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
    bench = LidDrivenCavityQualitative(reynolds=100, reference="ghia")
    profile = bench.profile_reference_np()
    batch = {
        "xy_f": torch.rand(8, 2),
        "xy_bc": torch.rand(8, 2),
        "xy_data": torch.empty(0, 2),
        "targets_data": bench.exact_torch(torch.empty(0, 2)),
        "xy_profile_u": torch.tensor(profile["u_xy"], dtype=torch.float32),
        "target_profile_u": torch.tensor(profile["u_ref"], dtype=torch.float32),
        "xy_profile_v": torch.tensor(profile["v_xy"], dtype=torch.float32),
        "target_profile_v": torch.tensor(profile["v_ref"], dtype=torch.float32),
    }
    batch["xy_profile_all"] = torch.cat([batch["xy_profile_u"], batch["xy_profile_v"]], dim=0)
    losses = compute_pointwise_losses(model, batch, bench, steady=True)
    assert losses["u_profile"].shape == (17, 1)
    assert losses["v_profile"].shape == (17, 1)
    assert losses["profile"].shape == (34, 1)


def test_cavity_profile_diagnostic_maps_are_nonzero_on_grid():
    model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
    bench = LidDrivenCavityQualitative(reynolds=100, reference="ghia")
    _, _, coords = bench.grid(16, 16)
    maps = DiagnosticMapBuilder(model, bench, torch.device("cpu")).build(coords, mode="residual_only")
    assert "u_profile_error" in maps
    assert "v_profile_error" in maps
    assert np.max(maps["profile_error"]) > 0.0
