"""Run one ablation mode."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.vara_trainer import VARATrainer
from src.utils.config import deep_update, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ablation", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    ablation_path = ROOT / "configs" / f"ablation_{args.ablation}.yaml"
    if ablation_path.exists():
        config = deep_update(config, load_config(ablation_path))
        mode = config.get("mode", args.ablation)
    else:
        mode = args.ablation
    trainer = VARATrainer(config, mode=mode)
    trainer.run()
    print(f"Run directory: {trainer.run_dir}")


if __name__ == "__main__":
    main()

