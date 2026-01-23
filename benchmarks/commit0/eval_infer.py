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

from benchmarks.utils.laminar import LaminarService
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
        "total_instances": 10,              # Number of REFERENCE repositories (not all 16!)
        "submitted_instances": 10,
        "completed_instances": 10,
        "resolved_instances": 5,            # Repos with all tests passed
        "unresolved_instances": 5,
        "empty_patch_instances": 0,
        "error_instances": 0,
        "total_tests": 3273,                # Sum of num_tests (REFERENCE REPOS ONLY)
        "total_passed_tests": 2977,         # Sum of num_passed (REFERENCE REPOS ONLY)
        "sum_num_passed": 2977,             # For accuracy calculation
        "completed_ids": [...],
        "resolved_ids": [...],
        "unresolved_ids": [...]
    }
    
    Note:
        The commit0-lite benchmark contains 16 instances total, but only 10 are used
        as reference (gold) instances for accuracy calculation on the leaderboard.
        
        Issue: PR #351 showed 100.7% accuracy because we were including all 16 repos
        instead of just the 10 reference repos, leading to incorrect test totals.
        
        References:
        - Leaderboard: https://commit-0.github.io/analysis/
        - Breakdown: https://commit-0.github.io/analysis_commit0_lite_reference/
        
        The 10 reference repos are: simpy, tinydb, marshmallow, wcwidth, imapclient,
        voluptuous, jinja, deprecated, cookiecutter, cachetools
    """
    logger.info(f"Processing {input_file} to generate report: {output_file}")

    # The 10 reference (gold) repos used for leaderboard scoring
    # See: https://commit-0.github.io/analysis_commit0_lite_reference/
    REFERENCE_REPOS = {
        "simpy",
        "tinydb",
        "marshmallow",
        "wcwidth",
        "imapclient",
        "voluptuous",
        "jinja",
        "deprecated",
        "cookiecutter",
        "cachetools",
    }

    completed_ids = []
    resolved_ids = []
    unresolved_ids = []
    total_tests = 0
    total_passed_tests = 0
    skipped_non_reference = []

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
                    logger.warning(f"Line {line_num}: Missing instance_id")
                    continue

                # Skip non-reference repos (only count the 10 reference repos)
                if instance_id not in REFERENCE_REPOS:
                    skipped_non_reference.append(instance_id)
                    logger.debug(
                        f"Skipping non-reference instance: {instance_id} (not in reference set)"
                    )
                    continue

                # Extract eval_result from test_result
                test_result = data.get("test_result", {})
                eval_result = test_result.get("eval_result", {})

                if not eval_result:
                    logger.warning(
                        f"Line {line_num}: Missing eval_result for {instance_id}"
                    )
                    continue

                # Extract metrics
                passed = eval_result.get("passed", 0)
                num_tests = eval_result.get("num_tests", 0)
                num_passed = eval_result.get("num_passed", 0)

                # Add to completed instances
                completed_ids.append(instance_id)

                # Count total tests and passed tests
                total_tests += num_tests
                total_passed_tests += num_passed

                # Determine if resolved (passed == 1.0 means all tests passed)
                if passed == 1.0:
                    resolved_ids.append(instance_id)
                else:
                    unresolved_ids.append(instance_id)

            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON - {e}")
            except Exception as e:
                logger.error(f"Line {line_num}: Unexpected error - {e}")

    # Log skipped non-reference instances
    if skipped_non_reference:
        logger.info(
            f"Skipped {len(skipped_non_reference)} non-reference instances: {sorted(set(skipped_non_reference))}"
        )

    # Generate report
    report = {
        "model_name_or_path": model_name,
        "total_instances": 10,  # Number of REFERENCE repositories (not all 16!)
        "submitted_instances": len(completed_ids),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": 0,  # Always 0 as per requirement
        "error_instances": 0,  # Always 0 as per requirement
        "total_tests": total_tests,
        "total_passed_tests": total_passed_tests,
        "sum_num_passed": total_passed_tests,  # Sum of num_passed across all instances (for accuracy calculation)
        "completed_ids": completed_ids,
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
    }

    # Write report
    with open(output_file, "w") as outfile:
        json.dump(report, outfile, indent=4)

    logger.info("Report generated successfully:")
    logger.info(f"  Total instances (REFERENCE repos): {report['total_instances']}")
    logger.info(f"  Completed instances: {report['completed_instances']}")
    logger.info(f"  Resolved instances (all tests passed): {report['resolved_instances']}")
    logger.info(f"  Unresolved instances: {report['unresolved_instances']}")
    logger.info(f"  Total tests across all instances: {report['total_tests']}")
    logger.info(f"  Sum of passed tests (sum_num_passed): {report['sum_num_passed']}")
    logger.info(f"  Note: Final accuracy will be calculated as sum_num_passed / 3628 total tests")


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

    # Determine output file (same name as input with .report.json extension)
    output_file = input_file.with_suffix(".report.json")

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Model name: {args.model_name}")

    try:
        # Process results and generate report
        process_commit0_results(str(input_file), str(output_file), args.model_name)

        # Update Laminar datapoints with evaluation scores
        LaminarService.get().update_evaluation_scores(str(input_file), str(output_file))

        # Generate cost report as final step
        generate_cost_report(str(input_file))

        logger.info("Script completed successfully!")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
