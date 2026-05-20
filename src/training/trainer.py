"""Shared trainer utilities."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.diagnostics import DiagnosticMapBuilder, PatchGrid, PatchScorer, WeakRegionDetector
from src.evaluation.metrics import evaluate_on_grid
from src.losses.base_losses import compute_global_losses, compute_pointwise_losses, weighted_sum
from src.losses.local_losses import compute_local_weighted_loss
from src.losses.pressure_losses import pressure_anchor_loss
from src.losses.vorticity_losses import vorticity_transport_residual
from src.models import build_mlp_from_config
from src.physics.kovasznay import KovasznayFlow
from src.physics.pressure_poisson import pressure_poisson_residual
from src.sampling import BoundarySampler, MixedAdaptiveSampler, UniformSampler
from src.training.checkpointing import save_checkpoint
from src.utils.config import save_config
from src.utils.device import get_device
from src.utils.io import ensure_dir, save_json
from src.utils.logging import CSVLogger, JSONListLogger, make_run_id
from src.utils.seed import set_seed
from src.visualization.controller_plots import save_intervention_timeline, save_patch_score_map
from src.visualization.fields import save_field_panel
from src.visualization.heatmaps import save_heatmap
from src.visualization.streamlines import save_streamlines


class ExperimentTrainer:
    """Reusable training scaffold for Kovasznay-first experiments."""

    def __init__(self, config: dict[str, Any], mode: str) -> None:
        self.config = config
        self.mode = mode
        self.seed = int(config.get("seed", 0))
        set_seed(self.seed, deterministic=bool(config.get("deterministic", True)))
        self.device = get_device(config.get("device", "auto"))
        self.benchmark = self._build_benchmark(config)
        self.steady = bool(config.get("pde", {}).get("steady", True))
        self.model = build_mlp_from_config(config, self.benchmark.bounds).to(self.device)
        optim_cfg = config.get("optimizer", {})
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(optim_cfg.get("lr", 1e-3)))
        self.global_step = 0
        self.best_score = math.inf

        patch_cfg = config.get("patches", {})
        self.patch_grid = PatchGrid(
            self.benchmark.bounds,
            nx_patches=int(patch_cfg.get("nx_patches", 4)),
            ny_patches=int(patch_cfg.get("ny_patches", 4)),
            nt_patches=int(patch_cfg.get("nt_patches", 1)),
        )
        diag_cfg = config.get("diagnostics", {})
        self.patch_scorer = PatchScorer(
            self.patch_grid,
            diagnostics=diag_cfg.get("variables"),
            aggregation=diag_cfg.get("aggregation", "mean"),
            normalization=diag_cfg.get("normalization", "percentile"),
            percentile=float(diag_cfg.get("aggregation_percentile", 90.0)),
            ema_rho=float(diag_cfg.get("ema_rho", 0.8)),
        )
        weak_cfg = config.get("weak_regions", {})
        self.weak_detector = WeakRegionDetector(
            percentile_threshold=float(weak_cfg.get("percentile_threshold", 80.0)),
            top_k_per_variable=int(weak_cfg.get("top_k_per_variable", 2)),
            min_active_patches=int(weak_cfg.get("min_active_patches", 1)),
            max_active_patches=int(weak_cfg.get("max_active_patches", 8)),
            persistence_cycles=int(weak_cfg.get("persistence_cycles", 1)),
        )

        exp_cfg = config.get("experiments", {})
        root = Path(exp_cfg.get("root", "experiments"))
        self.run_id = exp_cfg.get("run_id") or make_run_id(config.get("benchmark", "kovasznay"), mode, self.seed)
        self.run_dir = ensure_dir(root / "logs" / self.run_id)
        self.checkpoint_dir = ensure_dir(root / "checkpoints" / self.run_id)
        self.figure_dir = ensure_dir(root / "figures" / self.run_id)
        self.table_dir = ensure_dir(root / "tables" / self.run_id)
        self.metrics_dir = ensure_dir(root / "metrics" / self.run_id)
        save_config(config, self.run_dir / "config_snapshot.yaml")

        self.metrics_logger = CSVLogger(self.run_dir / "metrics.csv")
        self.loss_logger = CSVLogger(self.run_dir / "losses.csv")
        self.action_logger = JSONListLogger(self.run_dir / "action_log.json")
        self.weak_logger = JSONListLogger(self.run_dir / "weak_region_log.json")
        self.score_logger = JSONListLogger(self.run_dir / "patch_scores.json")
        self.accept_logger = JSONListLogger(self.run_dir / "acceptance_log.json")
        self.action_records: list[dict[str, Any]] = []

        self.uniform_sampler = UniformSampler(self.benchmark.bounds, self.device, self.seed)
        self.boundary_sampler = BoundarySampler(self.benchmark.bounds, self.device, self.seed + 1)
        sampler_cfg = config.get("sampling", {})
        self.adaptive_sampler = MixedAdaptiveSampler(
            self.benchmark.bounds,
            self.patch_grid,
            self.device,
            self.seed + 2,
            mixture=sampler_cfg.get("mixture"),
        )

    def _build_benchmark(self, config: dict[str, Any]) -> Any:
        name = config.get("benchmark", "kovasznay").lower()
        if name != "kovasznay":
            raise NotImplementedError(f"The first working target is Kovasznay; got {name}.")
        cfg = config.get("benchmark_params", {})
        return KovasznayFlow(
            reynolds=float(cfg.get("reynolds", 40.0)),
            x_min=float(cfg.get("x_min", -0.5)),
            x_max=float(cfg.get("x_max", 1.0)),
            y_min=float(cfg.get("y_min", -0.5)),
            y_max=float(cfg.get("y_max", 1.5)),
        )

    def initial_batch(self) -> dict[str, Any]:
        train_cfg = self.config.get("training", {})
        n_f = int(train_cfg.get("n_collocation", 1024))
        n_bc = int(train_cfg.get("n_boundary", 256))
        n_data = int(train_cfg.get("n_data", 256))
        xy_f = self.uniform_sampler.sample(n_f)
        xy_bc = self.boundary_sampler.sample(n_bc)
        xy_data = self.uniform_sampler.sample(n_data)
        return self.make_batch(xy_f, xy_bc, xy_data)

    def make_batch(self, xy_f: torch.Tensor, xy_bc: torch.Tensor, xy_data: torch.Tensor) -> dict[str, Any]:
        with torch.no_grad():
            targets = self.benchmark.exact_torch(xy_data)
        return {"xy_f": xy_f, "xy_bc": xy_bc, "xy_data": xy_data, "targets_data": targets}

    def validation_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config.get("validation", {})
        return self.benchmark.grid(int(cfg.get("nx", 50)), int(cfg.get("ny", 50)))

    def test_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config.get("test", {})
        return self.benchmark.grid(int(cfg.get("nx", 64)), int(cfg.get("ny", 64)))

    def train_epochs(
        self,
        batch: dict[str, Any],
        control_state: Any | None = None,
        cycle: int = 0,
        epochs_override: int | None = None,
        log_prefix: str = "",
    ) -> dict[str, float]:
        train_cfg = self.config.get("training", {})
        epochs = int(epochs_override if epochs_override is not None else train_cfg.get("epochs_per_cycle", 100))
        log_every = max(1, int(train_cfg.get("log_every", 25)))
        weights = dict(train_cfg.get("weights", {}))
        local_weights = {}
        active_aux_losses: set[str] = set()
        pressure_anchor_patches: dict[int, float] = {}
        if control_state is not None:
            weights = control_state.global_weights
            local_weights = control_state.local_weights
            active_aux_losses = control_state.active_aux_losses
            pressure_anchor_patches = control_state.pressure_anchor_patches

        last_losses: dict[str, float] = {}
        self.model.train()
        for local_epoch in range(epochs):
            self.optimizer.zero_grad(set_to_none=True)
            pointwise = compute_pointwise_losses(self.model, batch, self.benchmark, self.steady)
            if "pressure_poisson" in active_aux_losses:
                pointwise["pressure_poisson"] = pressure_poisson_residual(
                    self.model, batch["xy_f"], self.benchmark.nu
                ).pow(2)
            if "vorticity_transport" in active_aux_losses and not self.steady:
                pointwise["vorticity_transport"] = vorticity_transport_residual(
                    self.model, batch["xy_f"], self.benchmark.nu, steady=False
                ).pow(2)
            losses = compute_global_losses(pointwise)
            total = weighted_sum(losses, weights)
            local_loss, local_logs = compute_local_weighted_loss(
                pointwise,
                batch,
                self.patch_grid,
                local_weights,
                entropy_weight=float(self.config.get("controller", {}).get("entropy_weight", 0.0)),
            )
            total = total + local_loss
            if pressure_anchor_patches:
                patch_ids = self.patch_grid.assign_torch(batch["xy_f"])
                pred_f = self.model(batch["xy_f"])
                for pid, strength in pressure_anchor_patches.items():
                    mask = patch_ids == int(pid)
                    if torch.any(mask):
                        total = total + float(strength) * pressure_anchor_loss(pred_f[mask, 2:3], 0.0)
            total.backward()
            grad_norm = self._grad_norm()
            self.optimizer.step()

            last_losses = {k: float(v.detach().cpu()) for k, v in losses.items()}
            last_losses.update(local_logs)
            last_losses["total"] = float(total.detach().cpu())
            last_losses["grad_norm"] = grad_norm
            if local_epoch % log_every == 0 or local_epoch == epochs - 1:
                self.loss_logger.log({"cycle": cycle, "phase": log_prefix or "main", "epoch": self.global_step, **last_losses})
            self.global_step += 1
        return last_losses

    def _grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += float(torch.sum(p.grad.detach() ** 2).cpu())
        return float(math.sqrt(total))

    def diagnose(self) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], list[Any], np.ndarray, np.ndarray, np.ndarray]:
        X, Y, coords = self.validation_grid()
        builder = DiagnosticMapBuilder(self.model, self.benchmark, self.device, self.steady)
        maps = builder.build(coords, mode=self.config.get("diagnostics", {}).get("mode", "full_reference"))
        scores, names = self.patch_scorer.compute(maps, coords, update_ema=True)
        weak_regions = self.weak_detector.detect(scores, names, self.patch_grid)
        return maps, scores, names, weak_regions, X, Y, coords

    def resample_batch(
        self,
        batch: dict[str, Any],
        maps: dict[str, np.ndarray],
        coords: np.ndarray,
        weak_regions: list[Any],
        control_state: Any | None = None,
        adaptive: bool = True,
    ) -> dict[str, Any]:
        train_cfg = self.config.get("training", {})
        n_f = int(train_cfg.get("n_collocation", batch["xy_f"].shape[0]))
        n_bc = int(train_cfg.get("n_boundary", batch["xy_bc"].shape[0]))
        n_data = int(train_cfg.get("n_data", batch["xy_data"].shape[0]))
        if adaptive:
            priorities = control_state.sampling_priorities if control_state is not None else {}
            xy_f = self.adaptive_sampler.sample_interior(n_f, maps, coords, weak_regions, priorities)
            xy_data = self.adaptive_sampler.sample_interior(n_data, maps, coords, weak_regions, priorities)
        else:
            xy_f = self.uniform_sampler.sample(n_f)
            xy_data = self.uniform_sampler.sample(n_data)
        xy_bc = self.boundary_sampler.sample(n_bc)
        return self.make_batch(xy_f, xy_bc, xy_data)

    def evaluate_and_save_final(self) -> dict[str, float]:
        X, Y, coords = self.test_grid()
        metrics = evaluate_on_grid(self.model, self.benchmark, coords, self.device, self.steady)
        self.metrics_logger.log({"cycle": "final_test", **metrics})
        save_json(metrics, self.run_dir / "summary.json")
        pd.DataFrame([metrics]).to_csv(self.table_dir / "summary.csv", index=False)
        pd.DataFrame([metrics]).to_csv(self.run_dir / "summary_table.csv", index=False)
        self.save_plots(X, Y, coords)
        save_checkpoint(
            self.checkpoint_dir / "final.pt",
            self.model,
            self.optimizer,
            self.config,
            metrics,
            self.global_step,
            -1,
        )
        save_intervention_timeline(self.action_records, self.figure_dir / "intervention_timeline.png")
        return metrics

    def save_plots(self, X: np.ndarray, Y: np.ndarray, coords: np.ndarray) -> None:
        builder = DiagnosticMapBuilder(self.model, self.benchmark, self.device, self.steady)
        maps = builder.build(coords, mode="full_reference")
        shape = X.shape
        save_field_panel(
            X,
            Y,
            {
                "u pred": maps["u_pred"].reshape(shape),
                "v pred": maps["v_pred"].reshape(shape),
                "p pred centered": maps["p_pred"].reshape(shape),
                "omega pred": maps["omega_pred"].reshape(shape),
            },
            self.figure_dir / "predicted_fields.png",
        )
        save_field_panel(
            X,
            Y,
            {
                "u ref": maps["u_ref"].reshape(shape),
                "v ref": maps["v_ref"].reshape(shape),
                "p ref centered": maps["p_ref"].reshape(shape),
                "omega ref": maps["omega_ref"].reshape(shape),
            },
            self.figure_dir / "reference_fields.png",
        )
        for name in ["u_error", "v_error", "p_error_mean_centered", "omega_error", "pde_residual"]:
            save_heatmap(maps[name].reshape(shape), X, Y, self.figure_dir / f"{name}.png", name)
        save_streamlines(
            X,
            Y,
            maps["u_pred"].reshape(shape),
            maps["v_pred"].reshape(shape),
            self.figure_dir / "streamlines.png",
        )

    def maybe_checkpoint(self, cycle: int, metrics: dict[str, float]) -> None:
        score = (
            metrics.get("u_rel_l2", 0.0)
            + metrics.get("v_rel_l2", 0.0)
            + metrics.get("p_rel_l2_centered", 0.0)
            + metrics.get("omega_rel_l2", 0.0)
        )
        save_checkpoint(self.checkpoint_dir / "latest.pt", self.model, self.optimizer, self.config, metrics, self.global_step, cycle)
        if score < self.best_score:
            self.best_score = score
            save_checkpoint(self.checkpoint_dir / "best.pt", self.model, self.optimizer, self.config, metrics, self.global_step, cycle)
