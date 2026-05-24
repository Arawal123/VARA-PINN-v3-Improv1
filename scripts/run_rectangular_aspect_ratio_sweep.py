"""Run rectangular aspect-ratio benchmark sweeps."""

from benchmark_runner import parser_for, run_named_benchmark


def main() -> None:
    parser = parser_for("Rectangular aspect-ratio sweep")
    parser.add_argument("--aspect_ratios", nargs="+", type=float, default=[1.0, 2.0])
    args = parser.parse_args()
    for ratio in args.aspect_ratios:
        run_named_benchmark("rectangular_aspect_ratio", args, aspect_ratio=ratio)


if __name__ == "__main__":
    main()
