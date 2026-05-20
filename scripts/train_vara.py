"""Train a VARA-PINN experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.vara_trainer import VARATrainer
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="full_vara")
    args = parser.parse_args()
    config = load_config(args.config)
    trainer = VARATrainer(config, mode=args.mode)
    metrics = trainer.run()
    print(f"Run directory: {trainer.run_dir}")
    print("Final metrics:")
    for name in [
        "u_rel_l2",
        "v_rel_l2",
        "p_rel_l2_centered",
        "omega_rel_l2",
        "pde_residual_mean",
        "J_score",
    ]:
        if name in metrics:
            print(f"{name}={metrics[name]:.6e}")
    for name in ["accepted_interventions", "rejected_interventions", "rollback_count"]:
        if name in metrics:
            print(f"{name}={int(metrics[name])}")
    for name in [
        "number_of_active_patches",
        "most_frequently_targeted_variable",
        "most_frequently_targeted_patch",
    ]:
        if name in metrics:
            print(f"{name}={metrics[name]}")


if __name__ == "__main__":
    main()
