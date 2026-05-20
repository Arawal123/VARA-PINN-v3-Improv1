"""Run modes over multiple seeds."""

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
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    base = load_config(args.config)
    for seed in args.seeds:
        for mode in args.modes:
            config = dict(base)
            config["seed"] = seed
            trainer = VARATrainer(config, mode=mode)
            trainer.run()
            print(f"seed={seed} mode={mode}: {trainer.run_dir}")


if __name__ == "__main__":
    main()

