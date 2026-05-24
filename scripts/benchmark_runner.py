"""Shared helpers for lightweight professor benchmark runners."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.vara_trainer import VARATrainer
from src.utils.config import deep_update, load_config, save_config
from src.utils.io import save_json


METHOD_TO_MODE = {"vanilla": "vanilla_pinn", "vara": "local_constrained_vara"}


BENCHMARK_DEFAULTS: dict[str, dict[str, Any]] = {
    "channel_inflow_outflow": {
        "benchmark": "channel_inflow_outflow",
        "benchmark_params": {"reynolds": 40.0, "x_min": 0.0, "x_max": 2.0, "y_min": 0.0, "y_max": 1.0, "amplitude": 1.0},
        "diagnostics": {"mode": "full_reference"},
    },
    "lid_driven_cavity": {
        "benchmark": "lid_driven_cavity",
        "benchmark_params": {"reynolds": 100.0, "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "lid_velocity": 1.0},
        "diagnostics": {"mode": "residual_only"},
        "training": {"n_data": 0},
    },
    "double_vortex_box": {
        "benchmark": "double_vortex_box",
        "benchmark_params": {"reynolds": 40.0, "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "amplitude": 1.0},
        "diagnostics": {"mode": "full_reference"},
    },
    "boundary_condition_stress_test": {
        "benchmark": "boundary_condition_stress_test",
        "benchmark_params": {"reynolds": 40.0, "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "amplitude": 1.0},
        "diagnostics": {"mode": "full_reference"},
    },
    "rectangular_aspect_ratio": {
        "benchmark": "rectangular_aspect_ratio",
        "benchmark_params": {"reynolds": 40.0, "x_min": 0.0, "x_max": 2.0, "y_min": 0.0, "y_max": 1.0, "amplitude": 1.0},
        "diagnostics": {"mode": "full_reference"},
    },
}


def parser_for(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--method", choices=["vanilla", "vara", "both"], default="both")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--config", default="configs/kovasznay_debug.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="experiments")
    return parser


def run_named_benchmark(
    benchmark_name: str,
    args: argparse.Namespace,
    aspect_ratio: float | None = None,
) -> list[dict[str, Any]]:
    base = load_config(args.config)
    config = deep_update(base, BENCHMARK_DEFAULTS[benchmark_name])
    config["experiments"] = dict(config.get("experiments", {}))
    config["experiments"]["root"] = args.output_dir
    if args.device:
        config["device"] = args.device
    if aspect_ratio is not None:
        params = dict(config.get("benchmark_params", {}))
        params["x_min"] = 0.0
        params["y_min"] = 0.0
        params["y_max"] = 1.0
        params["x_max"] = float(aspect_ratio)
        config["benchmark_params"] = params
        config["benchmark"] = "rectangular_aspect_ratio"
    if config["benchmark"] == "lid_driven_cavity":
        config["training"] = {**config.get("training", {}), "n_data": 0}
    if args.quick:
        config = deep_update(
            config,
            {
                "model": {"hidden_layers": [32, 32]},
                "training": {
                    "adaptive_cycles": 1,
                    "epochs_per_cycle": 5,
                    "log_every": 5,
                    "n_collocation": 128,
                    "n_boundary": 64,
                    "n_data": 0 if config["benchmark"] == "lid_driven_cavity" else 64,
                },
                "local_controller": {"trial_epochs": 2, "max_actions_per_cycle": 2},
                "validation": {"nx": 16, "ny": 16},
                "test": {"nx": 20, "ny": 20},
            },
        )

    methods = ["vanilla", "vara"] if args.method == "both" else [args.method]
    rows = []
    for seed in args.seeds:
        for method in methods:
            run_config = deepcopy(config)
            run_config["seed"] = int(seed)
            mode = METHOD_TO_MODE[method]
            trainer = VARATrainer(run_config, mode=mode)
            metrics = trainer.run()
            metrics["benchmark"] = run_config["benchmark"]
            metrics["method"] = method
            metrics["mode"] = mode
            metrics["seed"] = int(seed)
            metrics["run_dir"] = str(trainer.run_dir)
            save_json(metrics, trainer.run_dir / "summary.json")
            pd.DataFrame([metrics]).to_csv(trainer.run_dir / "summary_table.csv", index=False)
            rows.append(metrics)
            print(f"benchmark={run_config['benchmark']} seed={seed} method={method}: {trainer.run_dir}")
    out = Path(args.output_dir) / "benchmark_runs"
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_ar{aspect_ratio:g}" if aspect_ratio is not None else ""
    pd.DataFrame(rows).to_csv(out / f"{benchmark_name}{suffix}_latest_results.csv", index=False)
    save_config(config, out / f"{benchmark_name}{suffix}_resolved_config.yaml")
    return rows
