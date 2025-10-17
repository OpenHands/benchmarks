"""
Critic system for evaluation.

This module contains the base Critic class and registry system for managing
different critics that evaluate whether an instance was successfully completed.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Set, Type

from pydantic import BaseModel

from benchmarks.utils.critics_utils import _has_non_empty_git_patch
from benchmarks.utils.models import EvalInstanceID, EvalOutput
from openhands.sdk import get_logger


logger = get_logger(__name__)


class Critic(ABC, BaseModel):
    """
    Base class for all critics.

    Critics evaluate whether an agent properly completed a task based on
    the evaluation output.
    """

    @abstractmethod
    def evaluate_instance(self, output: EvalOutput) -> bool:
        """
        Evaluate if an instance was successfully completed.

        Args:
            output: The evaluation output to check

        Returns:
            True if the instance was successfully completed, False otherwise
        """
        pass


class AgentFinishedCritic(Critic):
    """
    Default critic that evaluates whether an agent properly finished a task.

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
            if not _has_non_empty_git_patch(output):
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


class EmptyPatchCritic(Critic):
    """
    Critic that only evaluates whether a git patch is non-empty.

    This critic checks only one criterion:
    - The generated git patch is non-empty (actual changes were made)

    Unlike AgentFinishedCritic, this critic does not check for proper
    agent completion with FinishAction.
    """

    def evaluate_instance(self, output: EvalOutput) -> bool:
        """
        Evaluate if an instance has a non-empty git patch.

        Args:
            output: The evaluation output to check

        Returns:
            True if the git patch is non-empty, False otherwise
        """
        try:
            # Check if git patch is non-empty
            if not _has_non_empty_git_patch(output):
                logger.debug(f"Instance {output.instance_id}: Empty git patch")
                return False

            logger.debug(f"Instance {output.instance_id}: Non-empty git patch found")
            return True

        except Exception as e:
            logger.warning(f"Error evaluating instance {output.instance_id}: {e}")
            return False


class CriticRegistry:
    """
    Registry for managing available critics.

    This class provides a factory pattern for creating critics by name,
    making it easy to add new critics without modifying existing code.
    """

    _critics: Dict[str, Type[Critic]] = {}

    @classmethod
    def register(cls, name: str, critic_class: Type[Critic]) -> None:
        """
        Register a critic class with a given name.

        Args:
            name: The name to register the critic under
            critic_class: The critic class to register
        """
        cls._critics[name] = critic_class

    @classmethod
    def create_critic(cls, name: str) -> Critic:
        """
        Create a critic instance by name.

        Args:
            name: The name of the critic to create

        Returns:
            An instance of the requested critic

        Raises:
            ValueError: If the critic name is not registered
        """
        if name not in cls._critics:
            available = list(cls._critics.keys())
            raise ValueError(f"Unknown critic: {name}. Available critics: {available}")

        critic_class = cls._critics[name]
        return critic_class()

    @classmethod
    def list_critics(cls) -> list[str]:
        """
        Get a list of all registered critic names.

        Returns:
            List of registered critic names
        """
        return list(cls._critics.keys())


# Register default critics
CriticRegistry.register("finish_with_patch", AgentFinishedCritic)
CriticRegistry.register("empty_patch_critic", EmptyPatchCritic)


def get_failed_instances(
    output_file: str, critic: Optional[Critic] = None
) -> Set[EvalInstanceID]:
    """
    Get the set of failed instance IDs from an output file.

    Args:
        output_file: Path to the JSONL output file
        critic: Critic to use for evaluation. Must be provided.

    Returns:
        Set of instance IDs that failed

    Raises:
        ValueError: If critic is None
    """
    if critic is None:
        raise ValueError("Critic must be provided and cannot be None")

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

    logger.info(
        f"Found {len(failed_instances)} failed instances judged by critic in "
        f"{output_file}"
    )
    return failed_instances
