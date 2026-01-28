#!/usr/bin/env python3
"""
GAIA Evaluation Report Script

This script reads the output.jsonl produced by gaia run_infer, aggregates the
precomputed test_result.score values, optionally merges *_errors.jsonl to count
incomplete/error instances, and writes a SWE-bench-style summary report
(output.report.json next to the input file). It does not run model inference or
re-score answers; it only summarizes existing results and generates a cost
report.

Usage:
    uv run gaia-eval <path_to_output.jsonl>
"""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.gaia import constants
from benchmarks.utils.laminar import LaminarService
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


def process_gaia_results(
    input_file: str,
    output_file: str,
    model_name: str = constants.DEFAULT_MODEL_NAME,
) -> None:
    """
    Process GAIA output.jsonl and generate evaluation report.

    GAIA format:
    {
        "instance_id": "task_id",
        "test_result": {
            "score": true/false,
            "model_answer": "...",
            "model_answer_raw": "...",
            "ground_truth": "..."
        },
        "instruction": "...",
        "history": [...]
    }

    Report format (similar to SWE-Bench):
    {
        "model_name_or_path": "openhands",
        "total_instances": 165,
        "submitted_instances": 165,
        "completed_instances": 165,
        "incomplete_instances": 0,
        "resolved_instances": 100,
        "unresolved_instances": 65,
        "empty_patch_instances": 0,
        "error_instances": 0,
        "submitted_ids": [...],
        "completed_ids": [...],
        "incomplete_ids": [...],
        "resolved_ids": [...],
        "unresolved_ids": [...]
    }
    """
    logger.info(f"Processing {input_file} to generate report: {output_file}")

    completed_ids = []
    resolved_ids = []
    unresolved_ids = []
    incomplete_ids = []

    completed_seen = set()
    incomplete_seen = set()

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

                # Extract score from test_result
                test_result = data.get("test_result", {})
                score = test_result.get("score", False)

                if instance_id in completed_seen:
                    logger.warning(
                        f"Line {line_num}: Duplicate instance_id {instance_id}"
                    )
                    continue

                # Add to completed instances
                completed_ids.append(instance_id)
                completed_seen.add(instance_id)

                # Determine if resolved (score=True means correct answer)
                if score is True:
                    resolved_ids.append(instance_id)
                else:
                    unresolved_ids.append(instance_id)

            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON - {e}")
            except Exception as e:
                logger.error(f"Line {line_num}: Unexpected error - {e}")

    error_path = Path(input_file).with_name(f"{Path(input_file).stem}_errors.jsonl")
    if error_path.exists():
        with open(error_path, "r") as error_file:
            for line_num, line in enumerate(error_file, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)
                    instance_id = data.get("instance_id")
                    if not instance_id:
                        logger.warning(
                            f"Error file line {line_num}: Missing instance_id"
                        )
                        continue
                    if instance_id in completed_seen or instance_id in incomplete_seen:
                        logger.warning(
                            "Error file line %s: Duplicate instance_id %s",
                            line_num,
                            instance_id,
                        )
                        continue

                    incomplete_ids.append(instance_id)
                    incomplete_seen.add(instance_id)
                except json.JSONDecodeError as e:
                    logger.error(f"Error file line {line_num}: Invalid JSON - {e}")
                except Exception as e:
                    logger.error(f"Error file line {line_num}: Unexpected error - {e}")

    submitted_ids = completed_ids + incomplete_ids

    # Generate report
    report = {
        "model_name_or_path": model_name,
        "total_instances": len(submitted_ids),
        "submitted_instances": len(submitted_ids),
        "completed_instances": len(completed_ids),
        "incomplete_instances": len(incomplete_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": 0,
        "error_instances": len(incomplete_ids),
        "submitted_ids": submitted_ids,
        "completed_ids": completed_ids,
        "incomplete_ids": incomplete_ids,
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
    }

    # Write report
    with open(output_file, "w") as outfile:
        json.dump(report, outfile, indent=4)

    logger.info("Report generated successfully:")
    logger.info(f"  Total instances: {report['total_instances']}")
    logger.info(f"  Completed instances: {report['completed_instances']}")
    logger.info(f"  Resolved instances: {report['resolved_instances']}")
    logger.info(f"  Unresolved instances: {report['unresolved_instances']}")
    if report["completed_instances"] > 0:
        logger.info(
            f"  Success rate: {report['resolved_instances'] / report['completed_instances'] * 100:.1f}%"
        )


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process GAIA output and generate evaluation report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run gaia-eval output.jsonl
    uv run gaia-eval /path/to/output.jsonl --model-name "MyModel-v1.0"
        """,
    )

    parser.add_argument("input_file", help="Path to the GAIA output.jsonl file")

    parser.add_argument(
        "--model-name",
        default=constants.DEFAULT_MODEL_NAME,
        help=f"Model name to use in the model_name_or_path field (default: {constants.DEFAULT_MODEL_NAME})",
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
        process_gaia_results(
            str(input_file),
            str(output_file),
            args.model_name,
        )

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
