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
from src.losses.base_losses import compute_global_losses, compute_gpinn_gradient_penalty, compute_pointwise_losses, weighted_sum
from src.losses.local_losses import compute_local_weighted_loss
from src.losses.pressure_losses import pressure_anchor_loss
from src.losses.vorticity_losses import vorticity_transport_residual
from src.models import build_field_model_from_config
from src.physics.kovasznay import KovasznayFlow
from src.physics.cavity_reference import load_or_generate_full_field_reference
from src.physics.pressure_poisson import pressure_poisson_residual
from src.physics.rectangular_benchmarks import (
    BoundaryStressBoxFlow,
    DoubleVortexBoxFlow,
    LidDrivenCavityQualitative,
    PoiseuilleChannelFlow,
)
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
        self.dtype = self._resolve_dtype(config)
        self.model = build_field_model_from_config(config, self.benchmark.bounds).to(self.device).to(self.dtype)
        optim_cfg = config.get("optimizer", {})
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(optim_cfg.get("lr", 1e-3)))
        self.global_step = 0
        self.best_score = math.inf
        self.train_start_time = time.time()
        self.optimizer_steps = 0
        self.collocation_points_seen = 0
        self.boundary_points_seen = 0
        self.data_points_seen = 0
        self.lbfgs_steps = 0
        self.boundary_sample_counts = {"left": 0, "right": 0, "bottom": 0, "top": 0, "lid_corner_band": 0}

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
        self.last_losses: dict[str, float] = {}

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

    def _resolve_dtype(self, config: dict[str, Any]) -> torch.dtype:
        precision = str(config.get("precision", config.get("dtype", config.get("model", {}).get("dtype", "float32")))).lower()
        if precision in {"float32", "single", "fp32"}:
            return torch.float32
        if precision in {"float64", "double", "fp64"}:
            return torch.float64
        raise ValueError(f"Unsupported precision/dtype: {precision}")

    def _tensor(self, values: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.to(device=self.device, dtype=self.dtype)
        return torch.tensor(values, dtype=self.dtype, device=self.device)

    def _build_benchmark(self, config: dict[str, Any]) -> Any:
        name = config.get("benchmark", "kovasznay").lower()
        cfg = config.get("benchmark_params", {})
        common = {
            "reynolds": float(cfg.get("reynolds", 40.0)),
            "x_min": float(cfg.get("x_min", -0.5 if name == "kovasznay" else 0.0)),
            "x_max": float(cfg.get("x_max", 1.0)),
            "y_min": float(cfg.get("y_min", -0.5 if name == "kovasznay" else 0.0)),
            "y_max": float(cfg.get("y_max", 1.5 if name == "kovasznay" else 1.0)),
        }
        if name == "kovasznay":
            return KovasznayFlow(**common)
        rectangular = {**common, "amplitude": float(cfg.get("amplitude", 1.0))}
        if name in {"channel_inflow_outflow", "channel", "poiseuille"}:
            return PoiseuilleChannelFlow(**rectangular)
        if name in {"double_vortex_box", "double_vortex", "recirculating_vortex"}:
            return DoubleVortexBoxFlow(**rectangular)
        if name in {"boundary_condition_stress_test", "bc_stress"}:
            return BoundaryStressBoxFlow(**rectangular)
        if name in {"lid_driven_cavity", "cavity"}:
            full_field_reference_path = cfg.get("full_field_reference_path")
            generated_reference_validation = None
            if bool(cfg.get("generate_full_field_reference", False)) or str(full_field_reference_path).lower() == "auto":
                generated_path, generated_reference_validation = load_or_generate_full_field_reference(
                    reynolds=float(cfg.get("reynolds", 100.0)),
                    path=None,
                    nx=int(cfg.get("reference_nx", 65)),
                    ny=int(cfg.get("reference_ny", cfg.get("reference_nx", 65))),
                    max_iter=int(cfg.get("reference_max_iter", 4000)),
                    tolerance=float(cfg.get("reference_tolerance", 1e-6)),
                    force=bool(cfg.get("regenerate_reference", False)),
                )
                full_field_reference_path = str(generated_path)
            return LidDrivenCavityQualitative(
                **rectangular,
                lid_velocity=float(cfg.get("lid_velocity", 1.0)),
                reference=str(cfg.get("reference", "none")),
                reference_path=cfg.get("reference_path"),
                full_field_reference_path=full_field_reference_path,
                generated_reference_validation=generated_reference_validation,
                profile_only=bool(cfg.get("profile_only", full_field_reference_path is None)),
                has_reference=bool(full_field_reference_path) and not bool(cfg.get("profile_only", False)),
                reference_kind="full_field_cfd" if full_field_reference_path else str(cfg.get("reference", "residual_only")),
            )
        if name in {"rectangular_aspect_ratio", "rectangular_aspect_ratio_sweep"}:
            return PoiseuilleChannelFlow(**rectangular)
        raise NotImplementedError(f"Unknown benchmark: {name}.")

    def initial_batch(self) -> dict[str, Any]:
        train_cfg = self.config.get("training", {})
        n_f = int(train_cfg.get("n_collocation", 1024))
        n_bc = int(train_cfg.get("n_boundary", 256))
        n_data = int(train_cfg.get("n_data", 256))
        xy_f = self.uniform_sampler.sample(n_f)
        xy_bc = self._sample_boundary(n_bc)
        xy_data = self.uniform_sampler.sample(n_data)
        return self.make_batch(self._tensor(xy_f), xy_bc, self._tensor(xy_data))

    def _sample_boundary_numpy(self, n: int) -> np.ndarray:
        boundary_cfg = self.config.get("sampling", {}).get("cavity_boundary", {})
        if self._is_lid_driven_cavity() and bool(boundary_cfg.get("enabled", True)):
            pts = self.boundary_sampler.sample_lid_cavity_numpy(
                n,
                lid_fraction=float(boundary_cfg.get("lid_fraction", 0.45)),
                corner_fraction=float(boundary_cfg.get("corner_fraction", 0.25)),
                corner_width=float(boundary_cfg.get("corner_width", 0.12)),
                avoid_exact_corners=bool(boundary_cfg.get("avoid_exact_corners", True)),
                corner_epsilon=float(boundary_cfg.get("corner_epsilon", 1e-4)),
            )
        else:
            pts = self.boundary_sampler.sample_numpy(n)
        self._record_boundary_sampling(pts)
        return pts

    def _sample_boundary(self, n: int) -> torch.Tensor:
        return self._tensor(self._sample_boundary_numpy(n))

    def _record_boundary_sampling(self, pts: np.ndarray) -> None:
        if pts.size == 0:
            return
        x0, x1, y0, y1 = self.benchmark.bounds
        tol = 1e-8
        self.boundary_sample_counts["left"] += int(np.isclose(pts[:, 0], x0, atol=tol).sum())
        self.boundary_sample_counts["right"] += int(np.isclose(pts[:, 0], x1, atol=tol).sum())
        self.boundary_sample_counts["bottom"] += int(np.isclose(pts[:, 1], y0, atol=tol).sum())
        self.boundary_sample_counts["top"] += int(np.isclose(pts[:, 1], y1, atol=tol).sum())
        corner_width = 0.12 * min(max(x1 - x0, 1e-12), max(y1 - y0, 1e-12))
        near_lid_corner = (np.isclose(pts[:, 1], y1, atol=tol)) & (
            (pts[:, 0] <= x0 + corner_width) | (pts[:, 0] >= x1 - corner_width)
        )
        self.boundary_sample_counts["lid_corner_band"] += int(near_lid_corner.sum())

    def _is_lid_driven_cavity(self) -> bool:
        return str(self.config.get("benchmark", "")).lower() in {"lid_driven_cavity", "cavity"}

    def make_batch(self, xy_f: torch.Tensor, xy_bc: torch.Tensor, xy_data: torch.Tensor) -> dict[str, Any]:
        if (
            xy_data.shape[0] > 0
            and getattr(self.benchmark, "full_field_reference_path", None) is not None
            and bool(getattr(self.benchmark, "has_reference", False))
        ):
            target_np = self.benchmark.exact_np(xy_data.detach().cpu().numpy())
            targets = {name: self._tensor(value) for name, value in target_np.items()}
        else:
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
            if bool(self.config.get("baseline", {}).get("gpinn_gradient_penalty", False)):
                losses["gpinn"] = compute_gpinn_gradient_penalty(self.model, batch["xy_f"], self.benchmark, self.steady)
            total = weighted_sum(losses, weights)
            total = self._apply_self_adaptive_attention(total, losses)
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
            self._update_self_adaptive_attention(losses)

            last_losses = {k: float(v.detach().cpu()) for k, v in losses.items()}
            last_losses.update(local_logs)
            last_losses["total"] = float(total.detach().cpu())
            last_losses["grad_norm"] = grad_norm
            self.last_losses = dict(last_losses)
            if local_epoch % log_every == 0 or local_epoch == epochs - 1:
                self.loss_logger.log({"cycle": cycle, "phase": log_prefix or "main", "epoch": self.global_step, **last_losses})
            self.global_step += 1
            self.optimizer_steps += 1
            self.collocation_points_seen += int(batch["xy_f"].shape[0])
            self.boundary_points_seen += int(batch["xy_bc"].shape[0])
            self.data_points_seen += int(batch.get("xy_data", torch.zeros((0, 2), device=self.device)).shape[0])
        return last_losses

    def _apply_self_adaptive_attention(self, total: torch.Tensor, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        if not bool(self.config.get("baseline", {}).get("self_adaptive_attention", False)):
            return total
        if not hasattr(self, "attention_logits"):
            import torch.nn as nn

            keys = sorted(losses.keys())
            self.attention_logits = nn.ParameterDict(
                {key: nn.Parameter(torch.zeros((), device=self.device, dtype=self.dtype)) for key in keys}
            )
            lr = float(self.config.get("baseline", {}).get("attention_lr", self.config.get("optimizer", {}).get("lr", 1e-3)))
            self.attention_optimizer = torch.optim.Adam(self.attention_logits.parameters(), lr=lr)
        attention_total = total.new_tensor(0.0)
        for key, loss in losses.items():
            if key not in self.attention_logits:
                continue
            scale = torch.nn.functional.softplus(self.attention_logits[key]) + 1e-6
            attention_total = attention_total + scale * loss
        return attention_total

    def _update_self_adaptive_attention(self, losses: dict[str, torch.Tensor]) -> None:
        if not bool(self.config.get("baseline", {}).get("self_adaptive_attention", False)):
            return
        if not hasattr(self, "attention_optimizer"):
            return
        self.attention_optimizer.zero_grad(set_to_none=True)
        objective = next(iter(losses.values())).new_tensor(0.0)
        for key, loss in losses.items():
            if key not in self.attention_logits:
                continue
            scale = torch.nn.functional.softplus(self.attention_logits[key]) + 1e-6
            objective = objective - scale * loss.detach()
        objective.backward()
        self.attention_optimizer.step()
        with torch.no_grad():
            for param in self.attention_logits.values():
                param.clamp_(-6.0, 6.0)

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
        xy_bc = self._sample_boundary(n_bc)
        return self.make_batch(self._tensor(xy_f), xy_bc, self._tensor(xy_data))

    def evaluate_and_save_final(self) -> dict[str, float]:
        X, Y, coords = self.test_grid()
        metrics = evaluate_on_grid(self.model, self.benchmark, coords, self.device, self.steady)
        metrics.update(self._run_metadata())
        metrics["final_total_loss"] = float(self.last_losses.get("total", float("nan")))
        metrics["reference_kind"] = getattr(self.benchmark, "reference_kind", "analytical")
        metrics["has_reference"] = bool(getattr(self.benchmark, "has_reference", True))
        metrics["run_type"] = str(self.config.get("run_type", "full"))
        metrics["reportable"] = metrics["run_type"] != "smoke"
        metrics["collapse_evaluated"] = bool(metrics["reportable"])
        metrics["collapsed"] = self._collapsed(metrics)
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

    def maybe_run_lbfgs(self, batch: dict[str, Any], control_state: Any | None = None) -> None:
        lbfgs_cfg = self.config.get("optimizer", {}).get("lbfgs", {})
        if not bool(lbfgs_cfg.get("enabled", False)):
            return
        max_iter = int(lbfgs_cfg.get("max_iter", 100))
        if max_iter <= 0:
            return
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=float(lbfgs_cfg.get("lr", 1.0)),
            max_iter=max_iter,
            history_size=int(lbfgs_cfg.get("history_size", 50)),
            tolerance_grad=float(lbfgs_cfg.get("tolerance_grad", 1e-9)),
            tolerance_change=float(lbfgs_cfg.get("tolerance_change", 1e-12)),
            line_search_fn=lbfgs_cfg.get("line_search_fn", "strong_wolfe"),
        )
        weights = dict(self.config.get("training", {}).get("weights", {}))
        if control_state is not None:
            weights = control_state.global_weights

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            pointwise = compute_pointwise_losses(self.model, batch, self.benchmark, self.steady)
            losses = compute_global_losses(pointwise)
            if bool(self.config.get("baseline", {}).get("gpinn_gradient_penalty", False)):
                losses["gpinn"] = compute_gpinn_gradient_penalty(self.model, batch["xy_f"], self.benchmark, self.steady)
            total = weighted_sum(losses, weights)
            total.backward()
            self.last_losses = {k: float(v.detach().cpu()) for k, v in losses.items()}
            self.last_losses["total"] = float(total.detach().cpu())
            return total

        optimizer.step(closure)
        self.lbfgs_steps += max_iter
        self.optimizer_steps += max_iter
        self.collocation_points_seen += max_iter * int(batch["xy_f"].shape[0])
        self.boundary_points_seen += max_iter * int(batch["xy_bc"].shape[0])
        self.data_points_seen += max_iter * int(batch.get("xy_data", torch.zeros((0, 2), device=self.device)).shape[0])
        self.optimizer = optimizer

    def _run_metadata(self) -> dict[str, Any]:
        model_cfg = self.config.get("model", {})
        method = str(self.config.get("method", self.mode))
        total_boundary = max(1, sum(self.boundary_sample_counts.values()))
        metadata = {
            "mode": self.mode,
            "method": method,
            "seed": self.seed,
            "benchmark": str(self.config.get("benchmark", "unknown")),
            "architecture": self._architecture_label(),
            "model_kind": str(model_cfg.get("kind", getattr(self.model, "field_kind", "direct_uvp"))),
            "parameter_count": self._parameter_count(),
            "optimizer_name": str(self.config.get("optimizer", {}).get("name", "adam")),
            "optimizer_steps": int(self.optimizer_steps),
            "adam_steps": int(max(0, self.optimizer_steps - self.lbfgs_steps)),
            "lbfgs_steps": int(self.lbfgs_steps),
            "collocation_points_seen": int(self.collocation_points_seen),
            "boundary_points_seen": int(self.boundary_points_seen),
            "data_points_seen": int(self.data_points_seen),
            "wall_clock_train_sec": float(time.time() - self.train_start_time),
            "full_field_reference_used": bool(getattr(self.benchmark, "has_reference", False)),
            "profile_reference_used": bool(getattr(self.benchmark, "has_profile_reference", False)),
            "boundary_left_fraction": self.boundary_sample_counts["left"] / total_boundary,
            "boundary_right_fraction": self.boundary_sample_counts["right"] / total_boundary,
            "boundary_bottom_fraction": self.boundary_sample_counts["bottom"] / total_boundary,
            "boundary_top_fraction": self.boundary_sample_counts["top"] / total_boundary,
            "boundary_lid_corner_band_fraction": self.boundary_sample_counts["lid_corner_band"] / total_boundary,
        }
        validation = getattr(self.benchmark, "generated_reference_validation", None) or {}
        for key, value in validation.items():
            metadata[f"reference_{key}"] = value
        ablation = self.config.get("ablation", {})
        if ablation:
            metadata["ablation_mode"] = str(ablation.get("mode", ""))
            components = ablation.get("components", {})
            for name in ["V", "R", "S", "A", "B"]:
                if name in components:
                    metadata[f"ablation_component_{name}"] = bool(components[name])
            metadata["replay_schedule_path"] = str(ablation.get("replay_schedule_path", ""))
        return metadata

    def _architecture_label(self) -> str:
        model_cfg = self.config.get("model", {})
        kind = model_cfg.get("kind", getattr(self.model, "field_kind", "direct_uvp"))
        return f"{kind}:{model_cfg.get('hidden_layers', [])}:{model_cfg.get('activation', 'tanh')}"

    def _parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def save_plots(self, X: np.ndarray, Y: np.ndarray, coords: np.ndarray) -> None:
        builder = DiagnosticMapBuilder(self.model, self.benchmark, self.device, self.steady)
        diag_mode = "full_reference" if getattr(self.benchmark, "has_reference", True) else "residual_only"
        maps = builder.build(coords, mode=diag_mode)
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
        if getattr(self.benchmark, "has_reference", True):
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

    def _collapsed(self, metrics: dict[str, Any]) -> bool:
        if not bool(metrics.get("collapse_evaluated", True)):
            return False
        thresholds = self.config.get("collapse_thresholds", {})
        has_reference = bool(metrics.get("has_reference", True))
        relative_names = ["u_rel_l2", "v_rel_l2", "p_rel_l2_centered", "omega_rel_l2"]
        for name in relative_names:
            value = metrics.get(name)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(numeric):
                continue
            if not math.isfinite(numeric):
                return True
            if numeric > float(thresholds.get(name, 5.0)):
                return True
        if has_reference:
            for name in ["u_rmse", "v_rmse", "p_rmse_centered", "omega_rmse"]:
                value = metrics.get(name)
                if value is None:
                    continue
                numeric = float(value)
                if not math.isfinite(numeric):
                    return True
                if numeric > float(thresholds.get(name, 5.0)):
                    return True
        for name in [
            "pde_residual_mean",
            "continuity_residual_mean",
            "momentum_residual_mean",
            "unweighted_validation_loss",
        ]:
            value = metrics.get(name)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                return True
            if numeric > float(thresholds.get(name, 10.0)):
                return True
        bc = metrics.get("boundary_condition_error")
        if bc is not None and math.isfinite(float(bc)) and float(bc) > float(thresholds.get("boundary_condition_error", 1.0)):
            return True
        return False
