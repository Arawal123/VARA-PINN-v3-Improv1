"""Train a baseline PINN mode."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.baseline_trainer import BaselineTrainer
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="vanilla_pinn")
    args = parser.parse_args()
    config = load_config(args.config)
    trainer = BaselineTrainer(config, mode=args.mode)
    metrics = trainer.run()
    print(f"Run directory: {trainer.run_dir}")
    print(f"Final u_rel_l2={metrics['u_rel_l2']:.6e} p_rel_l2_centered={metrics['p_rel_l2_centered']:.6e}")


if __name__ == "__main__":
    main()

