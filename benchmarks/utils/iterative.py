"""
Iterative mode utilities for evaluation.

This module contains utilities for implementing iterative mode evaluation,
using SDK critics to determine if an instance succeeded.
"""

import json
import os
from typing import Set

from benchmarks.utils.critics import CriticBase, evaluate_output
from benchmarks.utils.models import EvalInstanceID, EvalOutput
from openhands.sdk import get_logger


logger = get_logger(__name__)


def _get_output_rank(critic: CriticBase, output: EvalOutput) -> int:
    """
    Get the rank of an output for aggregation purposes.
    Higher rank is better.

    Ranks:
    - 2: critic-successful (best)
    - 1: non-error/critic-fail
    - 0: error (worst)
    """
    if output.error:
        return 0
    if evaluate_output(critic, output):
        return 2
    return 1


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

    # Dictionary to store the best result for each instance
    best_results: dict[EvalInstanceID, EvalOutput] = {}

    # Work backwards from the last attempt to the first
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

                        instance_id = output.instance_id
                        output_rank = _get_output_rank(critic, output)

                        if instance_id not in best_results:
                            # First time seeing this instance
                            best_results[instance_id] = output
                        else:
                            # Replace if this output has a higher rank
                            current_best = best_results[instance_id]
                            current_rank = _get_output_rank(critic, current_best)
                            if output_rank > current_rank:
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

    # Write the aggregated results
    final_path = os.path.join(output_dir, final_output_file)
    if not best_results:
        logger.warning("No results found to aggregate - creating empty output file")
    logger.info(f"Writing {len(best_results)} aggregated results to {final_path}")

    try:
        successful_count = 0
        with open(final_path, "w", encoding="utf-8") as f:
            for output in best_results.values():
                if not output.error:  # Skip outputs with errors
                    f.write(output.model_dump_json() + "\n")
                    successful_count += 1

        logger.info(
            f"Successfully wrote {successful_count} successful results to {final_path}"
        )

    except Exception as e:
        logger.error(f"Error writing aggregated results to {final_path}: {e}")
        raise
