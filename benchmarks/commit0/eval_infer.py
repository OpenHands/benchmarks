#!/usr/bin/env python3
"""
Commit0 Evaluation Script

This script processes OpenHands output.jsonl format for Commit0 benchmark
and generates a report similar to SWE-Bench format.

Usage:
    uv run commit0-eval <path_to_output.jsonl>
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from benchmarks.utils.models import load_output_file, select_best_attempts
from benchmarks.utils.report_costs import generate_cost_report


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    Report format (similar to SWE-Bench):
    {
        "total_instances": 16,
        "submitted_instances": 16,
        "completed_instances": 16,
        "resolved_instances": 5,
        "unresolved_instances": 11,
        "empty_patch_instances": 0,
        "error_instances": 0,
        "total_tests": 500,
        "total_passed_tests": 400,
        "completed_ids": [...],
        "resolved_ids": [...],
        "unresolved_ids": [...]
    }
    """
    logger.info(f"Processing {input_file} to generate report: {output_file}")

    completed_ids = []
    resolved_ids = []
    unresolved_ids = []
    error_ids = []
    total_tests = 0
    total_passed_tests = 0

    outputs = load_output_file(input_file)
    best_attempts = select_best_attempts(outputs)

    for instance_id, out in best_attempts.items():
        eval_result = out.test_result.get("eval_result", {})
        num_tests = eval_result.get("num_tests", 0)
        num_passed = eval_result.get("num_passed", 0)

        completed_ids.append(instance_id)
        total_tests += num_tests
        total_passed_tests += num_passed

        if out.resolved:
            resolved_ids.append(instance_id)
        else:
            unresolved_ids.append(instance_id)
            if out.status == "error":
                error_ids.append(instance_id)

    # Generate report
    report = {
        "model_name_or_path": model_name,
        "total_instances": len(best_attempts),
        "submitted_instances": len(completed_ids),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": 0,  # Always 0 as per requirement
        "error_instances": len(error_ids),
        "total_tests": total_tests,
        "total_passed_tests": total_passed_tests,
        "completed_ids": completed_ids,
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
    }

    # Write report
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as outfile:
        json.dump(report, outfile, indent=4)

    logger.info("Report generated successfully:")
    logger.info(f"  Total instances: {report['total_instances']}")
    logger.info(f"  Completed instances: {report['completed_instances']}")
    logger.info(f"  Resolved instances: {report['resolved_instances']}")
    logger.info(f"  Unresolved instances: {report['unresolved_instances']}")
    logger.info(f"  Total tests: {report['total_tests']}")
    logger.info(f"  Total passed tests: {report['total_passed_tests']}")
    logger.info(
        f"  Success rate: {report['resolved_instances'] / report['completed_instances'] * 100:.1f}%"
    )


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
        help="Model name to use in the model_name_or_path field (default: openhands)",
    )

    args = parser.parse_args()

    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        sys.exit(1)

    if not input_file.suffix == ".jsonl":
        logger.warning(f"Input file does not have .jsonl extension: {input_file}")

    # Determine output file (always use default name)
    harness_dir = input_file.parent / "harness"
    output_file = harness_dir / "commit0_report.json"

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Model name: {args.model_name}")

    try:
        # Process results and generate report
        process_commit0_results(str(input_file), str(output_file), args.model_name)

        # Generate cost report as final step
        generate_cost_report(str(input_file))

        logger.info("Script completed successfully!")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
