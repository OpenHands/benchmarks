"""Simple critic creation from argparse arguments."""

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

from benchmarks.utils.critics import (
    AgentFinishedCritic,
    CriticBase,
    EmptyPatchCritic,
    PassCritic,
)
from openhands.sdk import get_logger


logger = get_logger(__name__)


def add_critic_args(parser: ArgumentParser) -> None:
    """Add critic-related arguments to argparse parser."""
    parser.add_argument(
        "--critic",
        type=str,
        default="pass",
        help="Critic to use: pass, finish_with_patch, empty_patch_critic, client",
    )
    parser.add_argument(
        "--critic-config",
        type=str,
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
    if critic_name == "pass":
        return PassCritic(**kwargs)

    elif critic_name == "finish_with_patch":
        return AgentFinishedCritic(**kwargs)

    elif critic_name == "empty_patch_critic":
        return EmptyPatchCritic(**kwargs)

    else:
        raise ValueError(
            f"Unknown critic: {critic_name}. "
            f"Available: pass, finish_with_patch, empty_patch_critic"
        )
