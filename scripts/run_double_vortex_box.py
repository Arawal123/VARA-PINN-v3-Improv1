"""Run the manufactured double-vortex box benchmark."""

from benchmark_runner import parser_for, run_named_benchmark


def main() -> None:
    args = parser_for("Double-vortex box benchmark").parse_args()
    run_named_benchmark("double_vortex_box", args)


if __name__ == "__main__":
    main()
