#!/usr/bin/env python3
"""
Script to calculate costs from JSONL evaluation output files.

This script processes JSONL files containing evaluation results and calculates:
1. Individual costs for each JSONL file (summing accumulated_cost from all lines)
2. Aggregated cost for critic files (excluding the main output.jsonl)
3. Saves a detailed cost report as cost_report.jsonl in the same directory

Usage:
    python report_costs.py <directory_path>

The script looks for files matching:
- output.jsonl (main output file)
- output.critic_attempt_*.jsonl (critic attempt files)

Output:
- Console report with detailed cost breakdown
- cost_report.jsonl file with structured cost data
- total_cost reflects real spend (sum of critic attempt files when present, otherwise main output)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_jsonl_file(file_path: Path) -> List[Optional[Dict]]:
    """Read a JSONL file and return list of JSON objects."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def extract_accumulated_cost(jsonl_data: List[Optional[Dict]]) -> float:
    """Sum the accumulated costs from each line in JSONL data."""
    if not jsonl_data:
        return 0.0

    total_cost = 0.0

    # Sum accumulated costs from each line
    for entry in jsonl_data:
        # Skip None entries that can occur from null JSON values
        if entry is None:
            continue
        metrics = entry.get("metrics", {})
        accumulated_cost = metrics.get("accumulated_cost", 0.0)
        if accumulated_cost is not None:
            total_cost += float(accumulated_cost)

    return total_cost


def format_duration(seconds: float) -> str:
    """Format duration in seconds to mm:ss format."""
    minutes = int(seconds // 60)
    seconds_remainder = int(seconds % 60)
    return f"{minutes:02d}:{seconds_remainder:02d}"


def calculate_line_duration(entry: Optional[Dict]) -> Optional[float]:
    """Calculate the duration for a single line (entry) in seconds."""
    # Skip None entries that can occur from null JSON values
    if entry is None:
        return None
    history = entry.get("history", [])
    if not history:
        return None

    timestamps = []
    for event in history:
        timestamp_str = event.get("timestamp")
        if timestamp_str:
            try:
                # Parse ISO format timestamp
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                timestamps.append(timestamp)
            except ValueError:
                continue

    if len(timestamps) < 2:
        return None

    # Calculate duration from oldest to newest timestamp
    oldest = min(timestamps)
    newest = max(timestamps)
    duration = (newest - oldest).total_seconds()

    return duration


def calculate_time_statistics(jsonl_data: List[Optional[Dict]]) -> Dict:
    """Calculate time statistics for all lines in JSONL data."""
    if not jsonl_data:
        return {
            "average_duration": 0.0,
            "total_duration": 0.0,
            "total_lines": 0,
            "lines_with_duration": 0,
        }

    durations = []
    for entry in jsonl_data:
        duration = calculate_line_duration(entry)
        if duration is not None:
            durations.append(duration)

    if not durations:
        return {
            "average_duration": 0.0,
            "total_duration": 0.0,
            "total_lines": len(jsonl_data),
            "lines_with_duration": 0,
        }

    return {
        "average_duration": sum(durations) / len(durations),
        "total_duration": sum(durations),
        "total_lines": len(jsonl_data),
        "lines_with_duration": len(durations),
    }


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

    # Initialize data structures for JSON report
    report_data = {
        "directory": str(directory_path),
        "timestamp": datetime.now().isoformat(),
        "main_output": None,
        "critic_files": [],
        "summary": {},
    }

    main_cost: Optional[float] = None
    main_total_duration: Optional[float] = None

    # Process main output file
    if output_file:
        print("\nSelected instance in Main output.jsonl only:")
        print(f"  {output_file.name}")

        jsonl_data = read_jsonl_file(output_file)
        cost = extract_accumulated_cost(jsonl_data)
        time_stats = calculate_time_statistics(jsonl_data)
        main_cost = cost
        main_total_duration = time_stats.get("total_duration", 0.0)

        print(f"    Lines: {len(jsonl_data)}")
        print(f"    Cost: ${cost:.6f}")
        print("    Time Stats:")
        print(
            f"      Average Duration: {format_duration(time_stats['average_duration'])}"
        )
        print(
            f"      Lines with Duration: {time_stats['lines_with_duration']}/{time_stats['total_lines']}"
        )

        # Add to report data
        report_data["main_output"] = {
            "filename": output_file.name,
            "lines": len(jsonl_data),
            "cost": cost,
            "time_statistics": time_stats,
        }

    # Process critic files individually
    critic_total_cost = 0.0
    critic_total_duration = 0.0
    if critic_files:
        print("\nCritic Attempt Files:")

        for critic_file in critic_files:
            print(f"  {critic_file.name}")

            jsonl_data = read_jsonl_file(critic_file)
            cost = extract_accumulated_cost(jsonl_data)
            time_stats = calculate_time_statistics(jsonl_data)
            critic_total_cost += cost
            critic_total_duration += time_stats.get("total_duration", 0.0)

            print(f"    Lines: {len(jsonl_data)}")
            print(f"    Cost: ${cost:.6f}")
            print("    Time Stats:")
            print(
                f"      Average Duration: {format_duration(time_stats['average_duration'])}"
            )
            print(
                f"      Lines with Duration: {time_stats['lines_with_duration']}/{time_stats['total_lines']}"
            )

            # Add to report data
            report_data["critic_files"].append(
                {
                    "filename": critic_file.name,
                    "lines": len(jsonl_data),
                    "cost": cost,
                    "time_statistics": time_stats,
                }
            )

        print(f"\n  Total Critic Files Cost: ${critic_total_cost:.6f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")

    # Total cost represents actual spend:
    # - If critic files exist, they contain all attempts; use their sum.
    # - Otherwise, fall back to the main output cost.
    total_cost = critic_total_cost if critic_files else (main_cost or 0.0)

    # Total duration represents total time across all instances:
    # - If critic files exist, use their sum.
    # - Otherwise, fall back to the main output duration.
    total_duration = (
        critic_total_duration if critic_files else (main_total_duration or 0.0)
    )

    if main_cost is not None:
        print(f"  Main Output Cost (best results): ${main_cost:.6f}")
    if critic_files:
        print(f"  Sum Critic Files (all attempts): ${critic_total_cost:.6f}")
    print(f"  Total Cost (no double-count): ${total_cost:.6f}")

    summary = {"total_cost": total_cost, "total_duration": total_duration}

    if main_cost is not None:
        summary["only_main_output_cost"] = main_cost
    if critic_files:
        summary["sum_critic_files"] = critic_total_cost

    report_data["summary"] = summary

    # Save JSON report
    report_file = directory / "cost_report.jsonl"
    try:
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n📊 Cost report saved to: {report_file}")
    except Exception as e:
        print(f"\n⚠️  Warning: Could not save cost report to {report_file}: {e}")


def generate_cost_report(input_file: str) -> None:
    """
    Generate cost report for the evaluation directory.

    This function is designed to be called from other evaluation scripts
    to automatically generate cost reports after evaluation completion.

    Args:
        input_file: Path to the input output.jsonl file
    """
    try:
        from pathlib import Path

        input_path = Path(input_file)
        directory = input_path.parent

        # Use the calculate_costs function to generate the report
        calculate_costs(str(directory))

    except Exception as e:
        # Don't fail the entire script if cost reporting fails
        # Just print a warning and continue
        print(f"Warning: Failed to generate cost report: {e}", file=sys.stderr)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate costs from JSONL evaluation output files and save detailed report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script processes JSONL files and generates:
1. Console output with detailed cost breakdown
2. cost_report.jsonl file with structured cost data in the same directory

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
