"""Run the boundary-condition stress benchmark."""

from benchmark_runner import parser_for, run_named_benchmark


def main() -> None:
    args = parser_for("Boundary-condition stress benchmark").parse_args()
    run_named_benchmark("boundary_condition_stress_test", args)


if __name__ == "__main__":
    main()
