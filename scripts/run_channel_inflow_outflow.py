"""Run the channel inflow/outflow benchmark."""

from benchmark_runner import parser_for, run_named_benchmark


def main() -> None:
    args = parser_for("Channel inflow/outflow benchmark").parse_args()
    run_named_benchmark("channel_inflow_outflow", args)


if __name__ == "__main__":
    main()
