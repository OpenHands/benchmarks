"""Command-line entrypoint for adapter-backed reliability runs."""

import argparse
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import cast

from benchmarks.reliability.reporting import (
    write_scorecard_json,
    write_scorecard_markdown,
)
from benchmarks.reliability.runner import ReliabilityAdapter, run_scenario
from benchmarks.reliability.scenario import load_scenario


def main() -> None:
    """Run one scenario through a user-selected benchmark adapter."""
    parser = argparse.ArgumentParser(
        description="Run a deterministic fault-injection reliability scenario."
    )
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        required=True,
        help="Zero-argument adapter factory in module:attribute form.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    adapter = _load_adapter(args.adapter)
    result = run_scenario(
        scenario,
        adapter=adapter,
        output_root=args.output_dir,
    )
    write_scorecard_json((result,), args.output_dir / "scorecard.json")
    write_scorecard_markdown((result,), args.output_dir / "scorecard.md")


def _load_adapter(spec: str) -> ReliabilityAdapter:
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("adapter must use module:attribute form")
    module = importlib.import_module(module_name)
    factory_value = getattr(module, attribute_name)
    if not callable(factory_value):
        raise TypeError(f"{spec} is not callable")
    factory = cast(Callable[[], ReliabilityAdapter], factory_value)
    return factory()


if __name__ == "__main__":
    main()
