"""
Critic system for evaluation.

This files provides utility functions
for working with EvalOutput in benchmarks and CriticBase implementations.
"""

import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Set

from benchmarks.utils.models import EvalInstanceID, EvalOutput
from openhands.sdk import get_logger
from openhands.sdk.critic import (
    AgentFinishedCritic,
    APIBasedCritic,
    CriticBase,
    EmptyPatchCritic,
    PassCritic,
)


logger = get_logger(__name__)


CRITIC_NAME_TO_CLASS = {
    "pass": PassCritic,
    "finish_with_patch": AgentFinishedCritic,
    "empty_patch_critic": EmptyPatchCritic,
    "api": APIBasedCritic,
}


def add_critic_args(parser: ArgumentParser) -> None:
    """Add critic-related arguments to argparse parser."""
    parser.add_argument(
        "--critic",
        type=str,
        default="pass",
        help=(
            "Name of the critic to use for evaluation (default: 'pass'). "
            "Critics determine whether an agent's output is considered successful "
            "and whether another attempt should be made in iterative evaluation mode. "
            "Available critics: "
            "'pass' - Always accepts the output (no retry logic, suitable for single-attempt runs), "
            "'finish_with_patch' - Requires both AgentFinishAction and non-empty git patch, "
            "'empty_patch_critic' - Only requires non-empty git patch. "
            "For single-attempt runs (default), 'pass' is recommended as the actual evaluation "
            "is performed by the benchmark's own scoring system."
        ),
    )
    parser.add_argument(
        "--critic-config",
        type=str,
        default=None,
        help="Path to JSON config file with critic parameters (e.g., {'api_key': 'xyz', 'timeout': 120})",
    )


def create_critic(args: Namespace) -> CriticBase:
    """
    Create a critic from parsed argparse arguments.

    Args:
        args: Parsed arguments from argparse

    Returns:
        Critic instance

    Example:
        # Simple critic
        parser = get_parser()
        args = parser.parse_args(['--critic', 'pass'])
        critic = create_critic(args)

        # Critic with config file
        args = parser.parse_args(['--critic', 'client', '--critic-config', 'critic.json'])
        critic = create_critic(args)
    """
    critic_name = args.critic

    # Load config if provided
    kwargs = {}
    if args.critic_config:
        config_path = Path(args.critic_config)
        if not config_path.exists():
            raise ValueError(f"Critic config file not found: {args.critic_config}")

        with open(config_path) as f:
            kwargs = json.load(f)

        logger.info(
            f"Loaded critic config from {args.critic_config}: {list(kwargs.keys())}"
        )

    # Create critic (Pydantic will validate the kwargs)
    if critic_name in CRITIC_NAME_TO_CLASS:
        critic_class = CRITIC_NAME_TO_CLASS[critic_name]
        critic = critic_class(**kwargs)
        logger.info(f"Created critic: {critic_name} with args: {kwargs}")
        return critic
    else:
        raise ValueError(
            f"Unknown critic: {critic_name}. "
            f"Available: pass, finish_with_patch, empty_patch_critic"
        )


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


def get_failed_instances(output_file: str) -> Set[EvalInstanceID]:
    """
    Get the set of failed instance IDs from an output file.

    Args:
        output_file: Path to the JSONL output file

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

                    # Check critic result (already set during evaluation)
                    if not output.critic_result.success:
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
