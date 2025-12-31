#!/usr/bin/env python3
"""
Commit0 Evaluation Script

This script processes OpenHands output.jsonl format for Commit0 benchmark
and generates a SWE-Bench-compatible report.

Usage:
    uv run commit0-eval <path_to_output.jsonl>
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from benchmarks.swebench.report import SwebenchReport
from benchmarks.utils.report import write_report
from benchmarks.utils.report_costs import generate_cost_report


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Commit0InstanceMetrics(BaseModel):
    num_tests: int
    num_passed: int
    pass_rate: float


class Commit0Report(SwebenchReport):
    total_tests: int
    total_passed_tests: int
    instance_metrics: dict[str, Commit0InstanceMetrics] = Field(default_factory=dict)
    average_pass_rate: float | None = None


def process_commit0_results(
    input_file: str, output_file: str, model_name: str = "openhands"
) -> None:
    """
    Process Commit0 output.jsonl and generate evaluation report.

    Commit0 format:
    {
        "instance_id": "deprecated",
        "test_result": {
            "eval_result": {
                "name": "deprecated",
                "sum": 0.02629628915747162,
                "passed": 1.0,
                "num_passed": 171,
                "num_tests": 171
            }
        },
        "instruction": "...",
        "history": [...]
    }

    Report format (SWE-Bench compatible):
    {
        "total_instances": 16,
        "submitted_instances": 16,
        "completed_instances": 16,
        "resolved_instances": 5,
        "unresolved_instances": 11,
        "empty_patch_instances": 0,
        "error_instances": 0,
        "completed_ids": [...],
        "incomplete_ids": [...],
        "submitted_ids": [...],
        "resolved_ids": [...],
        "unresolved_ids": [...],
        "empty_patch_ids": [...],
        "error_ids": [...],
        "schema_version": 2,
        "unstopped_instances": 0,
        "unstopped_containers": [...],
        "unremoved_images": [...]
    }
    """
    logger.info("Processing %s to generate report: %s", input_file, output_file)

    completed_ids: list[str] = []
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    total_tests = 0
    total_passed_tests = 0
    instance_metrics: dict[str, Commit0InstanceMetrics] = {}
    pass_rates: list[float] = []

    with open(input_file, "r") as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                # Extract required fields
                instance_id = data.get("instance_id")
                if not instance_id:
                    logger.warning("Line %s: Missing instance_id", line_num)
                    continue

                # Extract eval_result from test_result
                test_result = data.get("test_result", {})
                eval_result = test_result.get("eval_result", {})

                if not eval_result:
                    logger.warning(
                        "Line %s: Missing eval_result for %s", line_num, instance_id
                    )
                    continue

                # Extract metrics
                passed = eval_result.get("passed")
                num_tests = eval_result.get("num_tests", 0)
                num_passed = eval_result.get("num_passed", 0)
                if passed is None:
                    passed = (num_passed / num_tests) if num_tests else 0.0

                # Add to completed instances
                completed_ids.append(instance_id)

                # Count total tests and passed tests
                total_tests += num_tests
                total_passed_tests += num_passed
                pass_rates.append(float(passed))

                instance_metrics[instance_id] = Commit0InstanceMetrics(
                    num_tests=num_tests,
                    num_passed=num_passed,
                    pass_rate=float(passed),
                )

                # Determine if resolved (passed == 1.0 means all tests passed)
                if passed == 1.0:
                    resolved_ids.append(instance_id)
                else:
                    unresolved_ids.append(instance_id)

            except json.JSONDecodeError as e:
                logger.error("Line %s: Invalid JSON - %s", line_num, e)
            except Exception as e:
                logger.error("Line %s: Unexpected error - %s", line_num, e)

    average_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else None

    report = Commit0Report(
        model_name_or_path=model_name,
        total_instances=16,  # Fixed as per requirement
        submitted_instances=len(completed_ids),
        completed_instances=len(completed_ids),
        resolved_instances=len(resolved_ids),
        unresolved_instances=len(unresolved_ids),
        empty_patch_instances=0,  # Always 0 as per requirement
        error_instances=0,  # Always 0 as per requirement
        total_tests=total_tests,
        total_passed_tests=total_passed_tests,
        completed_ids=completed_ids,
        submitted_ids=completed_ids,
        resolved_ids=resolved_ids,
        unresolved_ids=unresolved_ids,
        instance_metrics=instance_metrics,
        average_pass_rate=average_pass_rate,
    )

    write_report(Path(output_file), report)

    logger.info("Report generated successfully:")
    logger.info("  Total instances: %s", report.total_instances)
    logger.info("  Completed instances: %s", report.completed_instances)
    logger.info("  Resolved instances: %s", report.resolved_instances)
    logger.info("  Unresolved instances: %s", report.unresolved_instances)
    logger.info("  Total tests: %s", report.total_tests)
    logger.info("  Total passed tests: %s", report.total_passed_tests)
    if report.completed_instances:
        logger.info(
            "  Success rate: %.1f%%",
            report.resolved_instances / report.completed_instances * 100,
        )
    else:
        logger.info("  Success rate: N/A")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process Commit0 output and generate evaluation report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run commit0-eval output.jsonl
    uv run commit0-eval /path/to/output.jsonl --model-name "MyModel-v1.0"
        """,
    )

    parser.add_argument("input_file", help="Path to the Commit0 output.jsonl file")

    parser.add_argument(
        "--model-name",
        default="openhands",
        help="Model name for logging (default: openhands)",
    )

    args = parser.parse_args()

    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error("Input file does not exist: %s", input_file)
        sys.exit(1)

    if not input_file.suffix == ".jsonl":
        logger.warning("Input file does not have .jsonl extension: %s", input_file)

    # Determine output file (same name as input with .report.json extension)
    output_file = input_file.with_suffix(".report.json")

    logger.info("Input file: %s", input_file)
    logger.info("Output file: %s", output_file)
    logger.info("Model name: %s", args.model_name)

    try:
        # Process results and generate report
        process_commit0_results(str(input_file), str(output_file), args.model_name)

        # Generate cost report as final step
        generate_cost_report(str(input_file))

        logger.info("Script completed successfully!")

    except Exception as e:
        logger.error("Script failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
