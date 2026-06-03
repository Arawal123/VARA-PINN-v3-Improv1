from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_on_grid
from src.physics.cavity_reference import (
    generate_lid_cavity_full_field_reference,
    interpolate_full_field,
    load_full_field_reference,
    load_lid_cavity_profile_reference,
)
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


def test_full_field_reference_load_and_interpolation(tmp_path):
    path = tmp_path / "full_field.csv"
    rows = []
    for y in [0.0, 1.0]:
        for x in [0.0, 1.0]:
            rows.append({"x": x, "y": y, "u": x + y, "v": x - y, "p": x, "omega": y})
    pd.DataFrame(rows).to_csv(path, index=False)
    ref = load_full_field_reference(path)
    interp = interpolate_full_field(ref, np.array([[0.5, 0.5]]))
    assert np.allclose(interp["u"], [[1.0]])
    assert np.allclose(interp["v"], [[0.0]])


def test_generated_cavity_reference_shape_and_ghia_validation():
    ref, validation = generate_lid_cavity_full_field_reference(
        reynolds=100,
        nx=9,
        ny=9,
        max_iter=3,
        tolerance=0.0,
    )
    assert ref["x"].shape == (81,)
    assert ref["u"].shape == (81,)
    assert validation["solver_grid_nx"] == 9
    assert np.isfinite(validation["ghia_validation_score"])


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
