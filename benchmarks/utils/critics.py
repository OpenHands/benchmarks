"""
Critic system for evaluation.

This module re-exports SDK critics and provides utility functions
for working with EvalOutput in benchmarks.
"""

import json
import os
from typing import Set

from benchmarks.utils.models import EvalInstanceID, EvalOutput
from openhands.sdk import get_logger
from openhands.sdk.critic import (
    AgentFinishedCritic,
    CriticBase,
    CriticResult,
    EmptyPatchCritic,
    PassCritic,
)
from openhands.sdk.event import LLMConvertibleEvent


# Re-export SDK critic components for convenience
__all__ = [
    "CriticBase",
    "CriticResult",
    "AgentFinishedCritic",
    "EmptyPatchCritic",
    "PassCritic",
    "extract_git_patch",
    "evaluate_output",
    "get_completed_instances",
    "get_failed_instances",
]

logger = get_logger(__name__)


def extract_git_patch(eval_output: EvalOutput) -> str | None:
    """
    Extract git patch from EvalOutput.

    Args:
        eval_output: The evaluation output

    Returns:
        Git patch string or None if not present
    """
    if not eval_output.test_result:
        return None
    return eval_output.test_result.get("git_patch")


def evaluate_output(critic: CriticBase, eval_output: EvalOutput) -> bool:
    """
    Evaluate an EvalOutput using a critic.

    This is a convenience function that extracts history and git_patch
    from EvalOutput and calls the critic's evaluate method.

    Args:
        critic: The SDK critic to use
        eval_output: The evaluation output to check

    Returns:
        True if the instance was successfully completed, False otherwise
    """
    try:
        # Convert history to events
        events = eval_output.history
        llm_events: list[LLMConvertibleEvent] = [
            e for e in events if isinstance(e, LLMConvertibleEvent)
        ]

        # Extract git patch
        git_patch = extract_git_patch(eval_output)

        # Call the SDK critic
        result = critic.evaluate(llm_events, git_patch)

        return result.success

    except Exception as e:
        logger.warning(f"Error evaluating output {eval_output.instance_id}: {e}")
        return False


def get_completed_instances(output_file: str) -> Set[EvalInstanceID]:
    """
    Get all instance IDs present in output file
    (completed, regardless of success/failure).

    Args:
        output_file: Path to the JSONL output file

    Returns:
        Set of instance IDs that were completed (processed)
    """
    completed_instances: Set[EvalInstanceID] = set()

    if not os.path.exists(output_file):
        return completed_instances

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    output = EvalOutput.model_validate(data)
                    completed_instances.add(output.instance_id)

                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Invalid JSON on line {line_num} in {output_file}: {e}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error processing line {line_num} in {output_file}: {e}"
                    )

    except Exception as e:
        logger.warning(f"Error reading output file {output_file}: {e}")

    return completed_instances


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

    logger.info(
        f"Found {len(failed_instances)} failed instances judged by critic in "
        f"{output_file}"
    )
    return failed_instances
