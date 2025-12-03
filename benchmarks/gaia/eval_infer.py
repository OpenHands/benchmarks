#!/usr/bin/env python3
"""
GAIA Evaluation Script

This script processes OpenHands output.jsonl format and computes GAIA scores.

Usage:
    uv run gaia-eval <path_to_output.jsonl>
"""

import argparse
import json
import sys
from pathlib import Path

from openhands.sdk import get_logger


logger = get_logger(__name__)


def evaluate_gaia_results(input_file: str) -> dict[str, int | float]:
    """
    Evaluate GAIA results from OpenHands output.jsonl file.

    OpenHands format:
    {
        "instance_id": "task_id",
        "test_result": {
            "score": true/false
        },
        "metadata": {...},
        "instruction": "...",
        "error": null,
        "history": [...]
    }

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating GAIA results from {input_file}")

    total = 0
    success = 0
    errors = 0

    with open(input_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                # Extract required fields
                instance_id = data.get("instance_id")
                if not instance_id:
                    logger.warning(f"Line {line_num}: Missing instance_id")
                    errors += 1
                    continue

                # Extract score from test_result
                test_result = data.get("test_result", {})
                score = test_result.get("score", False)

                total += 1
                if score:
                    success += 1

            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON - {e}")
                errors += 1
            except Exception as e:
                logger.error(f"Line {line_num}: Unexpected error - {e}")
                errors += 1

    if total == 0:
        logger.error("No valid entries found in the file")
        raise ValueError("No valid entries were evaluated")

    success_rate = success / total if total > 0 else 0.0

    logger.info("Evaluation complete:")
    logger.info(f"  Total: {total}")
    logger.info(f"  Success: {success}")
    logger.info(f"  Success rate: {success_rate:.2%}")
    logger.info(f"  Errors: {errors}")

    return {
        "total": total,
        "success": success,
        "success_rate": success_rate,
        "errors": errors,
    }


def write_evaluation_report(
    input_file: str, metrics: dict[str, int | float], output_file: str | None = None
) -> None:
    """
    Write evaluation report to a JSON file.

    Args:
        input_file: Path to the input file
        metrics: Evaluation metrics
        output_file: Path to the output file (optional)
    """
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix(".report.json"))

    report = {
        "input_file": input_file,
        "metrics": metrics,
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report written to {output_file}")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Evaluate GAIA results from OpenHands output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run gaia-eval output.jsonl
    uv run gaia-eval /path/to/output.jsonl
    uv run gaia-eval output.jsonl --output-file report.json
        """,
    )

    parser.add_argument("input_file", help="Path to the OpenHands output.jsonl file")

    parser.add_argument(
        "--output-file",
        help=(
            "Output file for evaluation report "
            "(default: input_file with .report.json extension)"
        ),
    )

    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Only print metrics, skip writing report file",
    )

    args = parser.parse_args()

    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        sys.exit(1)

    if not input_file.suffix == ".jsonl":
        logger.warning(f"Input file does not have .jsonl extension: {input_file}")

    logger.info(f"Input file: {input_file}")

    try:
        # Evaluate results
        metrics = evaluate_gaia_results(str(input_file))

        # Print summary
        print("\n" + "=" * 80)
        print("GAIA Evaluation Results")
        print("=" * 80)
        print(f"Total instances: {metrics['total']}")
        print(f"Successful: {metrics['success']}")
        print(f"Success rate: {metrics['success_rate']:.2%}")
        if metrics["errors"] > 0:
            print(f"Errors: {metrics['errors']}")
        print("=" * 80 + "\n")

        if not args.skip_report:
            # Write report
            write_evaluation_report(str(input_file), metrics, args.output_file)

        logger.info("Script completed successfully!")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
