"""Export a compact metrics table from run directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="experiments/tables/summary.csv")
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        path = Path(run) / "summary_table.csv"
        if path.exists():
            row = pd.read_csv(path).iloc[0].to_dict()
            row["run"] = Path(run).name
            rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()

