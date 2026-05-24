"""Run all lightweight professor benchmarks."""

from benchmark_runner import parser_for, run_named_benchmark


def main() -> None:
    parser = parser_for("All professor benchmarks")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "channel_inflow_outflow",
            "lid_driven_cavity",
            "double_vortex_box",
            "boundary_condition_stress_test",
        ],
    )
    args = parser.parse_args()
    for benchmark in args.benchmarks:
        run_named_benchmark(benchmark, args)


if __name__ == "__main__":
    main()
