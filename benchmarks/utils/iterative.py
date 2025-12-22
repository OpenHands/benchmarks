"""
Iterative mode utilities for evaluation.

This module contains utilities for implementing iterative mode evaluation,
using SDK critics to determine if an instance succeeded.
"""

import json
import os
from pathlib import Path
from typing import Set

from benchmarks.utils.critics import CriticBase, evaluate_output
from benchmarks.utils.models import (
    EvalInstanceID,
    EvalOutput,
    cost_from_metrics,
    write_output_line,
)
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
    Aggregate results from multiple attempts into a final output file.

    Works backwards from the last attempt to the first, using the most recent
    successful attempt for each instance.

    Args:
        output_dir: Directory containing attempt files
        max_attempts: Maximum number of attempts
        critic: Critic instance to use for evaluation
        final_output_file: Name of the final output file
    """
    logger.info(f"Aggregating results from {max_attempts} attempts")

    best_results: dict[EvalInstanceID, EvalOutput] = {}

    for attempt in range(max_attempts, 0, -1):
        attempt_file = os.path.join(
            output_dir, f"output.critic_attempt_{attempt}.jsonl"
        )

        if not os.path.exists(attempt_file):
            logger.debug(f"Attempt file {attempt_file} does not exist, skipping")
            continue

        logger.info(f"Processing attempt {attempt}: {attempt_file}")

        try:
            with open(attempt_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        output = EvalOutput.model_validate(data)
                        if output.attempt is None:
                            output.attempt = attempt

                        instance_id = output.instance_id
                        is_successful = evaluate_output(critic, output)

                        if instance_id not in best_results:
                            best_results[instance_id] = output
                        elif is_successful:
                            current_best = best_results[instance_id]
                            current_is_successful = evaluate_output(
                                critic, current_best
                            )
                            if not current_is_successful:
                                best_results[instance_id] = output

                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Invalid JSON on line {line_num} in {attempt_file}: {e}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error processing line {line_num} in {attempt_file}: {e}"
                        )

        except Exception as e:
            logger.error(f"Error reading attempt file {attempt_file}: {e}")

    final_path = Path(output_dir) / final_output_file
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if final_path.exists():
        final_path.unlink()

    if not best_results:
        logger.warning("No results found to aggregate - creating empty output file")
        final_path.touch()
        return

    logger.info(f"Writing {len(best_results)} aggregated results to {final_path}")

    for output in best_results.values():
        if output.status is None:
            output.status = "error" if output.error else "success"
        if output.resolved is None and output.status != "error":
            output.resolved = evaluate_output(critic, output)
        if output.cost is None:
            output.cost = cost_from_metrics(output.metrics)
        if output.artifacts_url is None:
            output.artifacts_url = ""
        write_output_line(final_path, output)
