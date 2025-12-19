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
from typing import Dict, List, Optional, Tuple

from benchmarks.utils.output_schema import load_output_file


def extract_accumulated_cost(outputs) -> float:
    """Sum total_cost across standardized outputs."""
    total_cost = 0.0
    for entry in outputs:
        if entry.cost.total_cost is not None:
            total_cost += float(entry.cost.total_cost)
    return total_cost


def find_output_files(directory: Path) -> Tuple[Optional[Path], List[Path]]:
    """Find output.jsonl and critic attempt files in the directory."""
    output_file = None
    critic_files: List[Path] = []

    for file_path in directory.glob("*.jsonl"):
        if file_path.name == "output.jsonl":
            output_file = file_path
        elif file_path.name.startswith("output.critic_attempt_"):
            critic_files.append(file_path)

    critic_files.sort(key=lambda x: x.name)
    return output_file, critic_files


def calculate_costs(directory_path: str) -> None:
    """Calculate and report costs for all JSONL files in the directory."""
    directory = Path(directory_path)

    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist")
        sys.exit(1)

    if not directory.is_dir():
        print(f"Error: {directory_path} is not a directory")
        sys.exit(1)

    output_file, critic_files = find_output_files(directory)

    if not output_file and not critic_files:
        print(f"No output.jsonl or critic attempt files found in {directory_path}")
        sys.exit(1)

    print(f"Cost Report for: {directory_path}")
    print("=" * 80)

    report_data: Dict[str, object] = {
        "directory": str(directory_path),
        "timestamp": datetime.now().isoformat(),
        "main_output": None,
        "critic_files": [],
        "summary": {},
    }

    total_individual_costs = 0.0

    if output_file:
        print("\nSelected instance in Main output.jsonl only:")
        print(f"  {output_file.name}")

        jsonl_data = load_output_file(output_file)
        cost = extract_accumulated_cost(jsonl_data)
        total_individual_costs += cost

        print(f"    Lines: {len(jsonl_data)}")
        print(f"    Cost: ${cost:.6f}")

        report_data["main_output"] = {
            "file": str(output_file),
            "lines": len(jsonl_data),
            "cost": cost,
        }

    for critic_file in critic_files:
        print(f"\nIncluding critic file: {critic_file.name}")
        jsonl_data = load_output_file(critic_file)
        cost = extract_accumulated_cost(jsonl_data)
        total_individual_costs += cost

        print(f"    Lines: {len(jsonl_data)}")
        print(f"    Cost: ${cost:.6f}")

        report_data["critic_files"].append(
            {"file": str(critic_file), "lines": len(jsonl_data), "cost": cost}
        )

    report_data["summary"] = {
        "total_cost_all_files": total_individual_costs,
        "only_main_output_cost": extract_accumulated_cost(
            load_output_file(output_file)
        )
        if output_file
        else 0.0,
        "critic_only_cost": total_individual_costs
        - (
            extract_accumulated_cost(load_output_file(output_file))
            if output_file
            else 0.0
        ),
    }

    report_file = directory / "cost_report.jsonl"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("\nSummary:")
    print(f"  Total cost (all files): ${total_individual_costs:.6f}")
    print(
        f"  Main output cost: ${report_data['summary']['only_main_output_cost']:.6f}"
    )
    print(f"  Critic-only cost: ${report_data['summary']['critic_only_cost']:.6f}")
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
