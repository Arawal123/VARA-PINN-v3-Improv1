"""Aggregate reviewer-grade cavity results into publication-oriented tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_benchmark_results import (
    build_cavity_table,
    build_collapse,
    build_compute_budget,
    build_methodwise,
    build_paired_statistics_table,
    build_readable,
    build_seedwise,
    build_valid_seed_counts,
    collect_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate reviewer-grade cavity outputs.")
    parser.add_argument("--results_dir", default="experiments/logs")
    parser.add_argument("--output_dir", default="experiments/reviewer_summary")
    args = parser.parse_args()

    rows = collect_rows(Path(args.results_dir))
    if not rows:
        raise SystemExit(f"No summary.json files found under {args.results_dir}")
    import pandas as pd

    df = pd.DataFrame(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    main_full_field = build_methodwise(df)
    cavity_profile = build_cavity_table(main_full_field)
    paired = build_paired_statistics_table(df)
    seedwise = build_seedwise(df)
    compute_budget = build_compute_budget(df)
    collapse = build_collapse(df)
    valid_counts = build_valid_seed_counts(df)

    df.to_csv(out / "combined_reviewer_results.csv", index=False)
    main_full_field.to_csv(out / "main_full_field_table.csv", index=False)
    cavity_profile.to_csv(out / "cavity_profile_table.csv", index=False)
    paired.to_csv(out / "paired_improvement_statistics_table.csv", index=False)
    seedwise.to_csv(out / "seedwise_paired_improvement_table.csv", index=False)
    compute_budget.to_csv(out / "compute_budget_table.csv", index=False)
    collapse.to_csv(out / "collapse_failure_table.csv", index=False)
    valid_counts.to_csv(out / "valid_seed_counts_table.csv", index=False)

    build_readable(main_full_field).to_csv(out / "main_full_field_table_readable.csv", index=False)
    build_readable(cavity_profile).to_csv(out / "cavity_profile_table_readable.csv", index=False)
    build_readable(paired).to_csv(out / "paired_improvement_statistics_readable.csv", index=False)
    build_readable(compute_budget).to_csv(out / "compute_budget_readable.csv", index=False)
    build_readable(collapse).to_csv(out / "collapse_failure_readable.csv", index=False)
    print(f"Saved reviewer tables under {out}")


if __name__ == "__main__":
    main()

