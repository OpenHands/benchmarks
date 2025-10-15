"""
Iterative mode utilities for evaluation.

This module contains utilities for implementing iterative mode evaluation,
including the AgentFinishedCritic for determining if an instance succeeded.
"""

import json
import os
from typing import Dict, Optional, Set

from pydantic import BaseModel

from benchmarks.utils.models import EvalInstanceID, EvalOutput
from openhands.sdk import get_logger


logger = get_logger(__name__)


class AgentFinishedCritic(BaseModel):
    """
    Critic that evaluates whether an agent properly finished a task.

    This critic checks two main criteria:
    1. The agent's last action was an AgentFinishAction (proper completion)
    2. The generated git patch is non-empty (actual changes were made)
    """

    def evaluate_instance(self, output: EvalOutput) -> bool:
        """
        Evaluate if an instance was successfully completed.

        Args:
            output: The evaluation output to check

        Returns:
            True if the instance was successfully completed, False otherwise
        """
        try:
            # Check if git patch is non-empty
            if not self._has_non_empty_git_patch(output):
                logger.debug(f"Instance {output.instance_id}: Empty git patch")
                return False

            # Check if agent properly finished with AgentFinishAction
            if not self._has_agent_finish_action(output):
                logger.debug(f"Instance {output.instance_id}: No AgentFinishAction")
                return False

            logger.debug(f"Instance {output.instance_id}: Successfully completed")
            return True

        except Exception as e:
            logger.warning(f"Error evaluating instance {output.instance_id}: {e}")
            return False

    def _has_non_empty_git_patch(self, output: EvalOutput) -> bool:
        """Check if the git patch is non-empty."""
        git_patch = output.test_result.get("git_patch", "")
        return bool(git_patch and git_patch.strip())

    def _has_agent_finish_action(self, output: EvalOutput) -> bool:
        """Check if the last action was a FinishAction."""
        if not output.history:
            return False

        # Look for the last action in the history
        for event in reversed(output.history):
            if isinstance(event, dict) and event.get("kind") == "ActionEvent":
                action_kind = event.get("action", {}).get("kind", "")
                if action_kind == "FinishAction":
                    return True
                # If we find any other action type, the agent didn't finish properly
                elif action_kind:
                    return False

        return False


def get_failed_instances(
    output_file: str, critic: Optional[AgentFinishedCritic] = None
) -> Set[EvalInstanceID]:
    """
    Get the set of failed instance IDs from an output file.

    Args:
        output_file: Path to the JSONL output file
        critic: Optional critic to use for evaluation. If None, creates a default one.

    Returns:
        Set of instance IDs that failed
    """
    if critic is None:
        critic = AgentFinishedCritic()

    failed_instances: Set[EvalInstanceID] = set()

    if not os.path.exists(output_file):
        logger.warning(f"Output file {output_file} does not exist")
        return failed_instances

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    output = EvalOutput(**data)

                    if not critic.evaluate_instance(output):
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
    output_dir: str, max_attempts: int, final_output_file: str = "output.jsonl"
) -> None:
    """
    Aggregate results from multiple attempts into a final output file.

    Works backwards from the last attempt to the first, using the most recent
    successful attempt for each instance that has a non-empty git patch.

    Args:
        output_dir: Directory containing attempt files
        max_attempts: Maximum number of attempts
        final_output_file: Name of the final output file
    """
    logger.info(f"Aggregating results from {max_attempts} attempts")

    # Dictionary to store the best result for each instance
    best_results: Dict[EvalInstanceID, EvalOutput] = {}
    critic = AgentFinishedCritic()

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
                        output = EvalOutput(**data)

                        # Use this result if:
                        # 1. We haven't seen this instance yet, OR
                        # 2. This attempt is the first one to succeed
                        instance_id = output.instance_id
                        is_successful = critic._has_non_empty_git_patch(
                            output
                        ) and critic.evaluate_instance(output)

                        if instance_id not in best_results:
                            # First time seeing this instance
                            best_results[instance_id] = output
                        elif is_successful:
                            # This attempt succeeded, check if we should replace
                            current_best = best_results[instance_id]
                            current_is_successful = critic._has_non_empty_git_patch(
                                current_best
                            ) and critic.evaluate_instance(current_best)
                            if not current_is_successful:
                                # Replace failed result with successful one
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
        with open(final_path, "w", encoding="utf-8") as f:
            for output in best_results.values():
                f.write(output.model_dump_json() + "\n")

        logger.info(f"Successfully wrote aggregated results to {final_path}")

    except Exception as e:
        logger.error(f"Error writing aggregated results to {final_path}: {e}")
        raise
