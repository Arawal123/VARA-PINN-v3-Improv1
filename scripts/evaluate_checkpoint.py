"""Evaluate a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_on_grid
from src.models import build_mlp_from_config
from src.physics.kovasznay import KovasznayFlow
from src.utils.config import load_config
from src.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))
    params = config.get("benchmark_params", {})
    benchmark = KovasznayFlow(reynolds=float(params.get("reynolds", 40.0)))
    model = build_mlp_from_config(config, benchmark.bounds).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"])
    _, _, coords = benchmark.grid(config.get("test", {}).get("nx", 64), config.get("test", {}).get("ny", 64))
    print(evaluate_on_grid(model, benchmark, coords, device))


if __name__ == "__main__":
    main()

