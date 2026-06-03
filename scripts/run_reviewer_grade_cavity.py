"""Reviewer-grade lid-driven cavity experiment runner."""

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

from scripts.benchmark_runner import BASELINE_METHODS, METHOD_TO_MODE
from src.training.vara_trainer import VARATrainer
from src.utils.config import deep_update, load_config
from src.utils.io import save_json


ABLATION_SUITE: list[dict[str, Any]] = [
    {"method": "vanilla", "mode": "vanilla_pinn", "components": {"V": False, "R": False, "S": False, "A": False, "B": False}},
    {"method": "S_only", "mode": "residual_adaptive_sampling", "components": {"V": False, "R": False, "S": True, "A": False, "B": False}},
    {"method": "V_only", "mode": "local_vara", "components": {"V": True, "R": False, "S": False, "A": False, "B": False}},
    {"method": "R_plus_S", "mode": "local_vara", "components": {"V": False, "R": True, "S": True, "A": False, "B": False}},
    {"method": "V_plus_R", "mode": "local_vara", "components": {"V": True, "R": True, "S": False, "A": False, "B": False}},
    {"method": "V_plus_R_plus_S", "mode": "local_vara", "components": {"V": True, "R": True, "S": True, "A": False, "B": False}},
    {
        "method": "V_plus_R_plus_S_without_A",
        "mode": "local_vara",
        "components": {"V": True, "R": True, "S": True, "A": False, "B": True},
    },
    {"method": "full_vara", "mode": "local_constrained_vara", "components": {"V": True, "R": True, "S": True, "A": True, "B": True}},
    {
        "method": "full_vara_streamfunction",
        "mode": "local_constrained_vara",
        "model_kind": "streamfunction_p",
        "components": {"V": True, "R": True, "S": True, "A": True, "B": True},
    },
    {
        "method": "full_vara_without_streamfunction",
        "mode": "local_constrained_vara",
        "model_kind": "direct_uvp",
        "components": {"V": True, "R": True, "S": True, "A": True, "B": True},
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reviewer-grade lid-driven cavity suites.")
    parser.add_argument("--suite", choices=["smoke", "paper", "baseline", "ablation"], default="smoke")
    parser.add_argument("--config", default="configs/lid_driven_cavity_reviewer.yaml")
    parser.add_argument("--reynolds", type=float, default=100.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output_dir", default="experiments")
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=["float32", "float64"], default=None)
    parser.add_argument("--full_field_reference_path", default=None)
    parser.add_argument("--generate_reference", action="store_true")
    parser.add_argument("--replay_schedule_path", default=None)
    args = parser.parse_args()

    config = _load_config_with_optional_base(Path(args.config))
    config["benchmark"] = "lid_driven_cavity"
    config["experiments"] = {**config.get("experiments", {}), "root": args.output_dir}
    if args.device:
        config["device"] = args.device
    if args.precision:
        config["precision"] = args.precision
    params = dict(config.get("benchmark_params", {}))
    params["reynolds"] = float(args.reynolds)
    if args.full_field_reference_path:
        params["full_field_reference_path"] = args.full_field_reference_path
        params["profile_only"] = False
    if args.generate_reference:
        params["generate_full_field_reference"] = True
        params["full_field_reference_path"] = "auto"
        params["profile_only"] = False
    config["benchmark_params"] = params

    seeds = args.seeds
    if seeds is None:
        seeds = [0] if args.suite == "smoke" else list(range(10))
    if args.suite == "smoke":
        config = deep_update(config, _smoke_overrides())
        entries = [{"method": "vanilla", "mode": "vanilla_pinn"}, {"method": "vara", "mode": "local_constrained_vara"}]
    elif args.suite == "paper":
        entries = [
            {"method": "vanilla", "mode": "vanilla_pinn"},
            {"method": "vara_direct", "mode": "local_constrained_vara", "model_kind": "direct_uvp"},
            {"method": "vara_streamfunction", "mode": "local_constrained_vara", "model_kind": "streamfunction_p"},
        ]
    elif args.suite == "baseline":
        entries = [{"method": method, "mode": METHOD_TO_MODE[method]} for method in BASELINE_METHODS]
    else:
        entries = ABLATION_SUITE

    rows = []
    for seed in seeds:
        for entry in entries:
            run_config = deepcopy(config)
            run_config["seed"] = int(seed)
            run_config["run_type"] = args.suite
            run_config["method"] = entry["method"]
            if "model_kind" in entry:
                run_config["model"] = {**run_config.get("model", {}), "kind": entry["model_kind"]}
            if "components" in entry:
                run_config["ablation"] = {"mode": entry["method"], "components": entry["components"]}
                if args.replay_schedule_path:
                    run_config["ablation"]["replay_schedule_path"] = args.replay_schedule_path
                if not entry["components"].get("B", True):
                    sampling = dict(run_config.get("sampling", {}))
                    cavity = dict(sampling.get("cavity_boundary", {}))
                    cavity["enabled"] = False
                    sampling["cavity_boundary"] = cavity
                    run_config["sampling"] = sampling
            trainer = VARATrainer(run_config, mode=entry["mode"])
            metrics = trainer.run()
            metrics["method"] = entry["method"]
            metrics["mode"] = entry["mode"]
            metrics["seed"] = int(seed)
            metrics["suite"] = args.suite
            metrics["run_dir"] = str(trainer.run_dir)
            save_json(metrics, trainer.run_dir / "summary.json")
            pd.DataFrame([metrics]).to_csv(trainer.run_dir / "summary_table.csv", index=False)
            rows.append(metrics)
            print(f"suite={args.suite} Re={args.reynolds:g} seed={seed} method={entry['method']}: {trainer.run_dir}")

    out = Path(args.output_dir) / "reviewer_cavity_runs"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"lid_driven_cavity_re{int(args.reynolds)}_{args.suite}_latest_results.csv", index=False)


def _load_config_with_optional_base(path: Path) -> dict[str, Any]:
    cfg = load_config(path)
    base = cfg.get("base_config")
    if not base:
        return cfg
    base_path = (ROOT / base) if not Path(base).is_absolute() else Path(base)
    base_cfg = _load_config_with_optional_base(base_path)
    override = dict(cfg)
    override.pop("base_config", None)
    return deep_update(base_cfg, override)


def _smoke_overrides() -> dict[str, Any]:
    return {
        "precision": "float32",
        "run_type": "smoke",
        "benchmark_params": {
            "full_field_reference_path": None,
            "generate_full_field_reference": False,
            "profile_only": True,
            "reference_nx": 17,
            "reference_ny": 17,
            "reference_max_iter": 5,
        },
        "model": {"hidden_layers": [16, 16]},
        "optimizer": {"lbfgs": {"enabled": False}},
        "training": {
            "adaptive_cycles": 1,
            "epochs_per_cycle": 2,
            "log_every": 1,
            "n_collocation": 32,
            "n_boundary": 32,
            "n_data": 0,
        },
        "local_controller": {
            "trial_epochs": 1,
            "warmup_cycles": 0,
            "max_actions_per_cycle": 1,
            "rejection_recovery_epochs": 0,
        },
        "validation": {"nx": 8, "ny": 8},
        "test": {"nx": 9, "ny": 9},
        "patches": {"nx_patches": 2, "ny_patches": 2},
        "weak_regions": {"top_k_per_variable": 1, "max_active_patches": 2},
    }


if __name__ == "__main__":
    main()
