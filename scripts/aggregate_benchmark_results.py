"""Aggregate professor benchmark outputs into readable tables and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METRICS = [
    "u_rel_l2",
    "v_rel_l2",
    "p_rel_l2_centered",
    "omega_rel_l2",
    "u_rmse",
    "v_rmse",
    "p_rmse_centered",
    "omega_rmse",
    "pde_residual_mean",
    "continuity_residual_mean",
    "momentum_residual_mean",
    "boundary_condition_error",
    "final_total_loss",
    "J_score",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="experiments/logs")
    parser.add_argument("--output_dir", default="experiments/benchmark_summary")
    args = parser.parse_args()

    rows = collect_rows(Path(args.results_dir))
    if not rows:
        raise SystemExit(f"No summary.json files found under {args.results_dir}")
    df = pd.DataFrame(rows)
    out = Path(args.output_dir)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "combined_results.csv", index=False)
    methodwise = build_methodwise(df)
    methodwise.to_csv(out / "methodwise_mean_std_table.csv", index=False)
    methodwise.to_csv(out / "final_summary_table.csv", index=False)
    collapse = build_collapse(df)
    collapse.to_csv(out / "collapse_rate_table.csv", index=False)
    seedwise = build_seedwise(df)
    seedwise.to_csv(out / "seedwise_comparison_table.csv", index=False)
    save_bar_plots(methodwise, fig_dir)
    save_seedwise_scatter(df, fig_dir)

    print("Saved:")
    for name in [
        "combined_results.csv",
        "final_summary_table.csv",
        "collapse_rate_table.csv",
        "seedwise_comparison_table.csv",
        "methodwise_mean_std_table.csv",
    ]:
        print(f"  {out / name}")
    print(f"  {fig_dir}")


def collect_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*/summary.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cfg_path = path.parent / "config_snapshot.yaml"
        cfg = {}
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        mode = data.get("mode") or path.parent.name.split("_seed")[0].split("_", 1)[-1]
        row = {
            "run_id": path.parent.name,
            "run_dir": str(path.parent),
            "benchmark": data.get("benchmark", cfg.get("benchmark", "unknown")),
            "mode": data.get("mode", mode),
            "method": data.get("method", "vara" if "vara" in mode else "vanilla"),
            "seed": int(data.get("seed", cfg.get("seed", -1))),
            "reference_kind": data.get("reference_kind", "unknown"),
            "has_reference": data.get("has_reference", None),
            "accepted_interventions": data.get("accepted_interventions", 0),
            "rejected_interventions": data.get("rejected_interventions", 0),
            "rollback_count": data.get("rollback_count", 0),
            "number_of_active_patches": data.get("number_of_active_patches", 0),
            "most_frequently_targeted_variable": data.get("most_frequently_targeted_variable"),
            "most_frequently_targeted_patch": data.get("most_frequently_targeted_patch"),
            "collapsed": bool(data.get("collapsed", False)),
        }
        for metric in METRICS:
            row[metric] = data.get(metric, np.nan)
        rows.append(row)
    return rows


def build_methodwise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (benchmark, method), group in df.groupby(["benchmark", "method"]):
        row = {"benchmark": benchmark, "method": method, "n_runs": len(group)}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["benchmark", "method"])


def build_collapse(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (benchmark, method), group in df.groupby(["benchmark", "method"]):
        n = len(group)
        c = int(group["collapsed"].sum())
        rows.append({"benchmark": benchmark, "method": method, "collapse_rate_percent": 100.0 * c / n, "collapsed": c, "total": n})
    return pd.DataFrame(rows).sort_values(["benchmark", "method"])


def build_seedwise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (benchmark, seed), group in df.groupby(["benchmark", "seed"]):
        vanilla = group[group["method"] == "vanilla"]
        vara = group[group["method"] == "vara"]
        if vanilla.empty or vara.empty:
            continue
        v = vanilla.iloc[-1]
        a = vara.iloc[-1]
        row = {"benchmark": benchmark, "seed": seed}
        for metric in METRICS:
            v_val = pd.to_numeric(pd.Series([v[metric]]), errors="coerce").iloc[0]
            a_val = pd.to_numeric(pd.Series([a[metric]]), errors="coerce").iloc[0]
            row[f"{metric}_vanilla"] = v_val
            row[f"{metric}_vara"] = a_val
            row[f"{metric}_improvement_percent"] = (v_val - a_val) / v_val * 100.0 if pd.notna(v_val) and v_val != 0 and pd.notna(a_val) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["benchmark", "seed"]) if rows else pd.DataFrame()


def save_bar_plots(methodwise: pd.DataFrame, fig_dir: Path) -> None:
    for benchmark, group in methodwise.groupby("benchmark"):
        labels = []
        vanilla = []
        vara = []
        for metric in [
            "u_rmse",
            "v_rmse",
            "p_rmse_centered",
            "omega_rmse",
            "pde_residual_mean",
            "boundary_condition_error",
        ]:
            v = group[group["method"] == "vanilla"]
            a = group[group["method"] == "vara"]
            if v.empty or a.empty:
                continue
            labels.append(metric)
            vanilla.append(v.iloc[0][f"{metric}_mean"])
            vara.append(a.iloc[0][f"{metric}_mean"])
        if not labels:
            continue
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        ax.bar(x - 0.18, vanilla, 0.36, label="Vanilla")
        ax.bar(x + 0.18, vara, 0.36, label="VARA")
        ax.set_title(f"{benchmark}: VARA vs Vanilla")
        ax.set_ylabel("mean metric, lower is better")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(fig_dir / f"{benchmark}_vara_vs_vanilla_bar.png", dpi=220)
        plt.close(fig)


def save_seedwise_scatter(df: pd.DataFrame, fig_dir: Path) -> None:
    for benchmark, group in df.groupby("benchmark"):
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for method, sub in group.groupby("method"):
            ax.scatter(sub["seed"], pd.to_numeric(sub["pde_residual_mean"], errors="coerce"), label=method)
        ax.set_title(f"{benchmark}: seedwise PDE residual")
        ax.set_xlabel("seed")
        ax.set_ylabel("PDE residual mean")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.savefig(fig_dir / f"{benchmark}_seedwise_pde_residual.png", dpi=220)
        plt.close(fig)


if __name__ == "__main__":
    main()
