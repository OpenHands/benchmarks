#!/usr/bin/env python3
"""Terminal-Bench Evaluation Script.

This script processes Terminal-Bench output and generates evaluation reports.
It reads the output.jsonl produced by run_infer, aggregates results,
and writes a summary report.

Usage:
    uv run terminalbench-eval <path_to_output.jsonl>
"""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.utils.laminar import LaminarService
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


def process_terminalbench_results(
    input_file: str,
    output_file: str,
) -> dict:
    """Process Terminal-Bench output.jsonl and generate evaluation report.

    Terminal-Bench format (from harbor conversion):
    {
        "instance_id": "task_id",
        "test_result": {
            "trajectory_path": "...",
            "total_steps": N,
            "final_metrics": {...},
            "passed": true/false  # May be populated by harbor grading
        },
        "instruction": "...",
        "history": [...]
    }

    Report format (similar to SWE-Bench):
    {
        "total_instances": N,
        "submitted_instances": N,
        "completed_instances": N,
        "incomplete_instances": N,
        "resolved_instances": N,
        "unresolved_instances": N,
        "error_instances": N,
        "submitted_ids": [...],
        "completed_ids": [...],
        "incomplete_ids": [...],
        "resolved_ids": [...],
        "unresolved_ids": [...]
    }
    """
    logger.info(f"Processing {input_file} to generate report: {output_file}")

    completed_ids: list[str] = []
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    incomplete_ids: list[str] = []
    error_ids: list[str] = []

    completed_seen: set[str] = set()
    incomplete_seen: set[str] = set()

    # Aggregate metrics
    total_cost_usd = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    with open(input_file) as infile:
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

                if instance_id in completed_seen:
                    logger.warning(
                        f"Line {line_num}: Duplicate instance_id {instance_id}"
                    )
                    continue

                # Check for errors
                error = data.get("error")
                if error:
                    error_ids.append(instance_id)
                    incomplete_ids.append(instance_id)
                    incomplete_seen.add(instance_id)
                    continue

                # Extract test result
                test_result = data.get("test_result", {})

                # Check if task passed (harbor may include this)
                passed = test_result.get("passed")
                # If not explicitly set, we mark as completed but ungraded
                is_resolved = passed is True

                # Add to completed instances
                completed_ids.append(instance_id)
                completed_seen.add(instance_id)

                if is_resolved:
                    resolved_ids.append(instance_id)
                else:
                    unresolved_ids.append(instance_id)

                # Aggregate metrics
                metrics = data.get("metrics", {})
                final_metrics = test_result.get("final_metrics", {})

                cost = metrics.get("total_cost_usd") or final_metrics.get(
                    "total_cost_usd", 0.0
                )
                prompt_tokens = metrics.get("total_prompt_tokens") or final_metrics.get(
                    "total_prompt_tokens", 0
                )
                completion_tokens = metrics.get(
                    "total_completion_tokens"
                ) or final_metrics.get("total_completion_tokens", 0)

                total_cost_usd += cost or 0.0
                total_prompt_tokens += prompt_tokens or 0
                total_completion_tokens += completion_tokens or 0

            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON - {e}")
            except Exception as e:
                logger.error(f"Line {line_num}: Unexpected error - {e}")

    # Check for error file
    error_path = Path(input_file).with_name(f"{Path(input_file).stem}_errors.jsonl")
    if error_path.exists():
        with open(error_path) as error_file:
            for line_num, line in enumerate(error_file, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)
                    instance_id = data.get("instance_id")
                    if not instance_id:
                        continue
                    if instance_id in completed_seen or instance_id in incomplete_seen:
                        continue

                    incomplete_ids.append(instance_id)
                    incomplete_seen.add(instance_id)
                    error_ids.append(instance_id)
                except (json.JSONDecodeError, Exception) as e:
                    logger.error(f"Error file line {line_num}: {e}")

    submitted_ids = completed_ids + incomplete_ids

    # Generate report
    report = {
        "total_instances": len(submitted_ids),
        "submitted_instances": len(submitted_ids),
        "completed_instances": len(completed_ids),
        "incomplete_instances": len(incomplete_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "error_instances": len(error_ids),
        "submitted_ids": submitted_ids,
        "completed_ids": completed_ids,
        "incomplete_ids": incomplete_ids,
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "error_ids": error_ids,
        # Aggregate metrics
        "aggregate_metrics": {
            "total_cost_usd": total_cost_usd,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
        },
    }

    # Write report
    with open(output_file, "w") as outfile:
        json.dump(report, outfile, indent=4)

    logger.info("Report generated successfully:")
    logger.info(f"  Total instances: {report['total_instances']}")
    logger.info(f"  Completed instances: {report['completed_instances']}")
    logger.info(f"  Resolved instances: {report['resolved_instances']}")
    logger.info(f"  Unresolved instances: {report['unresolved_instances']}")
    logger.info(f"  Error instances: {report['error_instances']}")
    if report["completed_instances"] > 0:
        logger.info(
            f"  Success rate: "
            f"{report['resolved_instances'] / report['completed_instances'] * 100:.1f}%"
        )
    logger.info(f"  Total cost: ${total_cost_usd:.4f}")

    return report


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process Terminal-Bench output and generate evaluation report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run terminalbench-eval output.jsonl
    uv run terminalbench-eval /path/to/output.jsonl
        """,
    )

    parser.add_argument(
        "input_file", help="Path to the Terminal-Bench output.jsonl file"
    )
    parser.add_argument(
        "--output-file",
        help="Output file for report (default: input_file with .report.json extension)",
    )

    args = parser.parse_args()

    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        sys.exit(1)

    if not input_file.suffix == ".jsonl":
        logger.warning(f"Input file does not have .jsonl extension: {input_file}")

    # Determine output file
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = input_file.with_suffix(".report.json")

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")

    try:
        # Process results and generate report
        process_terminalbench_results(
            str(input_file),
            str(output_file),
        )

        # Update Laminar datapoints with evaluation scores
        LaminarService.get().update_evaluation_scores(str(input_file), str(output_file))

        # Generate cost report as final step
        generate_cost_report(str(input_file))

        logger.info("Script completed successfully!")
        print(json.dumps({"report_json": str(output_file)}))

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
