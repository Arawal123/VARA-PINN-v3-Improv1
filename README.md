# VARA-PINN

Variable-Aware Regional Adaptive Control for multi-field Navier-Stokes Physics-Informed Neural Networks.

This repository turns the original Kovasznay notebook prototype into a modular research codebase. The first fully wired target is Kovasznay flow with a working `full_vara` run: model training, diagnostics, patch scores, weak-region detection, local interventions, adaptive sampling, logs, plots, checkpoints, and final test metrics.

## Novelty Boundary

VARA-PINN is designed around variable-region pairs, not only global adaptive loss weighting or residual adaptive sampling. The controller diagnoses failures separately for velocity, pressure, pressure gradient, speed, vorticity, continuity, momentum residuals, aggregate PDE residual, and boundary violation. It then applies local interventions only in weak patches.

## Install

```bash
cd VARA-PINN
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quickstart

```bash
python scripts/train_vara.py --config configs/kovasznay.yaml --mode full_vara
```

The default config is intentionally small so the full path can be smoke-tested quickly. For paper runs, increase `epochs_per_cycle`, `adaptive_cycles`, model width/depth, point budgets, and seeds.

## Baselines and Ablations

```bash
python scripts/train_baseline.py --config configs/kovasznay.yaml --mode vanilla_pinn
python scripts/run_ablation.py --config configs/kovasznay.yaml --ablation full_vara
python scripts/run_multiseed.py --config configs/kovasznay.yaml --modes vanilla_pinn full_vara --seeds 0 1 2 3 4
```

Implemented modes share the same trainer surface. The first release prioritizes the Kovasznay full VARA path; the remaining modes are present as reproducible run modes and should be hardened with matched-compute studies before claims.

## Outputs

Each run writes:

- `experiments/logs/<run_id>/metrics.csv`
- `experiments/logs/<run_id>/losses.csv`
- `experiments/logs/<run_id>/action_log.json`
- `experiments/logs/<run_id>/weak_region_log.json`
- `experiments/logs/<run_id>/patch_scores.json`
- `experiments/checkpoints/<run_id>/best.pt`, `latest.pt`, `final.pt`
- `experiments/figures/<run_id>/*.png`
- `experiments/tables/<run_id>/summary.csv`

## Repository Structure

```text
configs/        benchmark and ablation configs
data/           raw, processed, and reference data placeholders
src/models/     MLP, multi-head MLP, Fourier features, SIREN
src/physics/    Navier-Stokes autograd residuals and benchmarks
src/losses/     global, local, pressure, vorticity losses
src/diagnostics/ diagnostic maps, patch scores, masks, weak regions
src/controllers/ VARA policies, actions, acceptance and rollback
src/sampling/   uniform, boundary, residual, regional, mixed sampling
src/training/   trainers, checkpointing, LBFGS helpers
src/evaluation/ metrics and regional analysis
src/visualization/ heatmaps, fields, controller plots
scripts/        command-line experiment entry points
tests/          unit tests for scientific plumbing
```

## Scientific Guardrails

The controller uses a validation diagnostic grid for adaptation. Final test metrics are computed separately at the end of training. Synthetic references may be used for validation diagnostics in controlled experiments, but claims should also include residual-only and sparse-data regimes.

## Current Limitations

Kovasznay is the first complete benchmark. Taylor-Green has an analytical reference skeleton, while lid-driven cavity and cylinder wake configs are prepared for external reference data integration. Reviewer-grade claims still require longer runs, broader baselines, multi-seed statistics, and matched point/compute budgets.

## Citation

```bibtex
@misc{vara_pinn_2026,
  title = {VARA-PINN: Variable-Aware Regional Adaptive Control for Multi-Field Navier-Stokes PINNs},
  author = {TBD},
  year = {2026},
  note = {Preprint in preparation}
}
```

