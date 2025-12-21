"""
Iterative mode utilities for evaluation.

This module contains utilities for implementing iterative mode evaluation,
using SDK critics to determine if an instance succeeded.
"""

import json
import os
from typing import Set

from benchmarks.utils.critics import CriticBase, evaluate_output
from benchmarks.utils.models import EvalInstanceID, EvalOutput, write_derived_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


def get_failed_instances(output_file: str, critic: CriticBase) -> Set[EvalInstanceID]:
    """
    Get the set of failed instance IDs from an output file.

    Args:
        output_file: Path to the JSONL output file
        critic: SDK critic to use for evaluation

    Returns:
        Set of instance IDs that failed
    """

    failed_instances: Set[EvalInstanceID] = set()

    if not os.path.exists(output_file):
        logger.warning(f"Output file {output_file} does not exist")
        return failed_instances

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    output = EvalOutput.model_validate(data)

                    # Evaluate using the critic
                    if not evaluate_output(critic, output):
                        failed_instances.add(output.instance_id)

                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Invalid JSON on line {line_num} in {output_file}: {e}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error processing line {line_num} in {output_file}: {e}"
                    )

    except Exception as e:
        logger.error(f"Error reading output file {output_file}: {e}")

    logger.info(f"Found {len(failed_instances)} failed instances in {output_file}")
    return failed_instances


def aggregate_results(
    output_dir: str,
    max_attempts: int,
    critic: "CriticBase",
    final_output_file: str = "output.jsonl",
) -> None:
    """
    Derive the summary report from standardized output.jsonl.

    Args:
        output_dir: Directory containing attempt files
        max_attempts: Maximum number of attempts
        critic: Critic instance to use for evaluation
        final_output_file: Name of the final output file
    """
    logger.info(f"Aggregating results from {max_attempts} attempts")

    output_path = os.path.join(output_dir, final_output_file)
    if not os.path.exists(output_path):
        logger.warning("Canonical output file missing: %s", output_path)
        return

    try:
        report_path = write_derived_report(output_dir)
        logger.info("Wrote derived report to %s", report_path)
    except Exception as e:
        logger.error("Error writing derived report: %s", e)
        raise
