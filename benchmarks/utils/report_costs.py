#!/usr/bin/env python3
"""
Cost reporting for standardized evaluation outputs.

Sums the `cost.total_cost` field from output.jsonl and per-attempt files,
and writes a lightweight summary to cost_report.json.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from benchmarks.utils.output_schema import load_output_file


def extract_accumulated_cost(outputs) -> float:
    """Sum total_cost across standardized outputs."""
    total_cost = 0.0
    for entry in outputs:
        if entry.cost.total_cost is not None:
            total_cost += float(entry.cost.total_cost)
    return total_cost


def find_output_file(directory: Path) -> Optional[Path]:
    """Find the canonical output.jsonl file in the directory."""
    output_file = directory / "output.jsonl"
    return output_file if output_file.exists() else None


def calculate_costs(directory_path: str) -> None:
    """Calculate and report costs for all JSONL files in the directory."""
    directory = Path(directory_path)

    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist")
        sys.exit(1)

    if not directory.is_dir():
        print(f"Error: {directory_path} is not a directory")
        sys.exit(1)

    output_file = find_output_file(directory)

    if not output_file:
        print(f"No output.jsonl found in {directory_path}")
        sys.exit(1)

    print(f"Cost Report for: {directory_path}")
    print("=" * 80)

    report_data: Dict[str, Any] = {
        "directory": str(directory_path),
        "timestamp": datetime.now().isoformat(),
        "main_output": None,
        "summary": {},
    }

    if output_file:
        print("\nSelected instance in Main output.jsonl only:")
        print(f"  {output_file.name}")

        jsonl_data = load_output_file(output_file)
        cost = extract_accumulated_cost(jsonl_data)

        print(f"    Lines: {len(jsonl_data)}")
        print(f"    Cost: ${cost:.6f}")

        report_data["main_output"] = {
            "file": str(output_file),
            "lines": len(jsonl_data),
            "cost": cost,
        }

    report_data["summary"] = {
        "total_cost_all_files": cost,
        "only_main_output_cost": cost,
        "critic_only_cost": 0.0,
    }

    report_file = directory / "cost_report.jsonl"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("\nSummary:")
    print(f"  Total cost (all files): ${cost:.6f}")
    print(f"  Main output cost: ${cost:.6f}")
    print("  Critic-only cost: $0.000000")
    print(f"\nDetailed cost report saved to: {report_file}")


def generate_cost_report(output_file: str) -> None:
    """Backwards-compatible wrapper used by eval scripts."""
    output_path = Path(output_file)
    target_dir = output_path.parent
    calculate_costs(str(target_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate costs from standardized evaluation outputs."
    )
    parser.add_argument(
        "directory",
        help="Directory containing output.jsonl and attempt files",
    )

    args = parser.parse_args()
    calculate_costs(args.directory)


if __name__ == "__main__":
    main()
