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

from pydantic import BaseModel, Field


class TimeStats(BaseModel):
    average: float
    maximum: float
    minimum: float
    measured_lines: int
    total_lines: int

    def as_minutes_seconds(self) -> Dict[str, str]:
        return {
            "average": format_duration(self.average),
            "max": format_duration(self.maximum),
            "min": format_duration(self.minimum),
        }


class FileCost(BaseModel):
    filename: str
    lines: int
    cost: float
    time: TimeStats


class DirectoryCostReport(BaseModel):
    directory: str
    timestamp: str
    main_output: Optional[FileCost] = None
    critic_files: List[FileCost] = Field(default_factory=list)
    total_cost: float
    critic_total: float
    main_cost: Optional[float]
    average_duration: float


def read_jsonl_file(file_path: Path) -> List[Dict]:
    """Read a JSONL file and return list of JSON objects."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


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


def format_duration(seconds: float) -> str:
    """Format duration in seconds to mm:ss format."""
    minutes = int(seconds // 60)
    seconds_remainder = int(seconds % 60)
    return f"{minutes:02d}:{seconds_remainder:02d}"


def calculate_line_duration(entry: Dict) -> Optional[float]:
    """Calculate the duration for a single line (entry) in seconds."""
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


def calculate_time_statistics(jsonl_data: List[Dict]) -> TimeStats:
    """Calculate time statistics for all lines in JSONL data."""
    durations = [
        duration
        for entry in jsonl_data
        if (duration := calculate_line_duration(entry)) is not None
    ]

    if not jsonl_data or not durations:
        return TimeStats(
            average=0.0,
            maximum=0.0,
            minimum=0.0,
            measured_lines=len(durations),
            total_lines=len(jsonl_data),
        )

    average = sum(durations) / len(durations)
    return TimeStats(
        average=average,
        maximum=max(durations),
        minimum=min(durations),
        measured_lines=len(durations),
        total_lines=len(jsonl_data),
    )


def find_output_files(directory: Path) -> Tuple[Optional[Path], List[Path]]:
    """Find output.jsonl and critic attempt files in the directory."""
    output_file = None
    critic_files = []

    for file_path in directory.glob("*.jsonl"):
        if file_path.name == "output.jsonl":
            output_file = file_path
        elif file_path.name.startswith("output.critic_attempt_"):
            critic_files.append(file_path)

    critic_files.sort(key=lambda x: x.name)

    return output_file, critic_files


def build_file_cost(file_path: Path) -> FileCost:
    data = read_jsonl_file(file_path)
    cost = extract_accumulated_cost(data)
    time_stats = calculate_time_statistics(data)
    return FileCost(
        filename=file_path.name,
        lines=len(data),
        cost=cost,
        time=time_stats,
    )


def render_file_report(label: str, report: FileCost) -> None:
    print(f"\n{label}:")
    print(f"  {report.filename}")
    print(f"    Lines: {report.lines}")
    print(f"    Cost: ${report.cost:.6f}")
    durations = report.time.as_minutes_seconds()
    print("    Time Stats:")
    print(f"      Average Duration: {durations['average']}")
    print(f"      Max Duration: {durations['max']}")
    print(f"      Min Duration: {durations['min']}")
    print(
        f"      Lines with Duration: {report.time.measured_lines}/{report.time.total_lines}"
    )


def calculate_costs(directory_path: str) -> None:
    """Calculate and report costs for all JSONL files in the directory."""
    directory = Path(directory_path)

    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist")
        sys.exit(1)

    if not directory.is_dir():
        print(f"Error: {directory_path} is not a directory")
        sys.exit(1)

    output_file, critic_paths = find_output_files(directory)
    if not output_file and not critic_paths:
        print(f"No output.jsonl or critic attempt files found in {directory_path}")
        sys.exit(1)

    print(f"Cost Report for: {directory_path}")
    print("=" * 80)

    main_report = build_file_cost(output_file) if output_file else None
    critic_reports = [build_file_cost(p) for p in critic_paths]
    critic_total = sum(r.cost for r in critic_reports)
    total_cost = critic_total if critic_reports else (main_report.cost if main_report else 0.0)
    average_duration = main_report.time.average if main_report else 0.0

    if main_report:
        render_file_report("Main output.jsonl", main_report)

    for report in critic_reports:
        render_file_report("Critic attempt", report)

    if critic_reports:
        print(f"\n  Total Critic Files Cost: ${critic_total:.6f}")

    print("\n" + "=" * 80)
    print("SUMMARY:")
    if main_report:
        print(f"  Main Output Cost (best results): ${main_report.cost:.6f}")
    if critic_reports:
        print(f"  Sum Critic Files (all attempts): ${critic_total:.6f}")
    print(f"  Total Cost (no double-count): ${total_cost:.6f}")

    report = DirectoryCostReport(
        directory=str(directory_path),
        timestamp=datetime.now().isoformat(),
        main_output=main_report,
        critic_files=critic_reports,
        total_cost=total_cost,
        critic_total=critic_total,
        main_cost=main_report.cost if main_report else None,
        average_duration=average_duration,
    )

    report_file = directory / "cost_report.jsonl"
    with open(report_file, "w") as f:
        json.dump(report.model_dump(), f, indent=2)
    print(f"\n📊 Cost report saved to: {report_file}")


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
