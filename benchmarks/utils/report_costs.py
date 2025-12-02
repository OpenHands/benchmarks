#!/usr/bin/env python3
"""
Script to calculate costs from JSONL evaluation output files.

This script processes JSONL files containing evaluation results and calculates:
1. Individual costs for each JSONL file (using the last accumulated cost from each line)
2. Aggregated cost for critic files (excluding the main output.jsonl)

Usage:
    python report_costs.py <directory_path>

The script looks for files matching:
- output.jsonl (main output file)
- output.critic_attempt_*.jsonl (critic attempt files)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_jsonl_file(file_path: Path) -> List[Dict]:
    """Read a JSONL file and return list of JSON objects."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def extract_accumulated_cost(jsonl_data: List[Dict]) -> float:
    """Sum the accumulated costs from each line in JSONL data."""
    if not jsonl_data:
        return 0.0

    total_cost = 0.0

    # Sum accumulated costs from each line
    for entry in jsonl_data:
        metrics = entry.get("metrics", {})
        accumulated_cost = metrics.get("accumulated_cost", 0.0)
        if accumulated_cost is not None:
            total_cost += float(accumulated_cost)

    return total_cost


def find_output_files(directory: Path) -> Tuple[Optional[Path], List[Path]]:
    """Find output.jsonl and critic attempt files in the directory."""
    output_file = None
    critic_files = []

    for file_path in directory.glob("*.jsonl"):
        if file_path.name == "output.jsonl":
            output_file = file_path
        elif file_path.name.startswith("output.critic_attempt_"):
            critic_files.append(file_path)

    # Sort critic files by attempt number
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

    # Find output files
    output_file, critic_files = find_output_files(directory)

    if not output_file and not critic_files:
        print(f"No output.jsonl or critic attempt files found in {directory_path}")
        sys.exit(1)

    print(f"Cost Report for: {directory_path}")
    print("=" * 80)

    total_individual_costs = 0.0

    # Process main output file
    if output_file:
        print("\nMain Output File:")
        print(f"  {output_file.name}")

        jsonl_data = read_jsonl_file(output_file)
        cost = extract_accumulated_cost(jsonl_data)
        total_individual_costs += cost

        print(f"    Lines: {len(jsonl_data)}")
        print(f"    Cost: ${cost:.6f}")

    # Process critic files individually
    if critic_files:
        print("\nCritic Attempt Files:")
        critic_total = 0.0

        for critic_file in critic_files:
            print(f"  {critic_file.name}")

            jsonl_data = read_jsonl_file(critic_file)
            cost = extract_accumulated_cost(jsonl_data)
            total_individual_costs += cost
            critic_total += cost

            print(f"    Lines: {len(jsonl_data)}")
            print(f"    Cost: ${cost:.6f}")

        print(f"\n  Total Critic Files Cost: ${critic_total:.6f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  Total Individual Files Cost: ${total_individual_costs:.6f}")

    if output_file and critic_files:
        # Calculate cost excluding main output.jsonl (only critic files)
        critic_only_total = total_individual_costs - extract_accumulated_cost(
            read_jsonl_file(output_file)
        )
        print(f"  Total Critic Files Only: ${critic_only_total:.6f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate costs from JSONL evaluation output files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python report_costs.py ./eval_outputs/my_experiment/
  python report_costs.py /path/to/evaluation/results/
        """,
    )

    parser.add_argument("directory", help="Directory containing JSONL output files")

    args = parser.parse_args()

    calculate_costs(args.directory)


if __name__ == "__main__":
    main()
