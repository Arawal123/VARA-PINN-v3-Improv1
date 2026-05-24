"""Run the residual-only lid-driven cavity benchmark."""

from benchmark_runner import parser_for, run_named_benchmark


def main() -> None:
    args = parser_for("Lid-driven cavity residual/boundary benchmark").parse_args()
    run_named_benchmark("lid_driven_cavity", args)


if __name__ == "__main__":
    main()
