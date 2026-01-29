"""
Argument parsing utilities for benchmarks.

This module defines common arguments used across all benchmarks.
Benchmark-specific defaults should be set via parser.set_defaults()
to match the evaluation repository configuration.
"""

import argparse

from benchmarks.utils.critics import add_critic_args


def get_parser(add_llm_config: bool = True) -> argparse.ArgumentParser:
    """Create and return argument parser without defaults.

    Each benchmark must call parser.set_defaults() before parse_args()
    to set values matching the evaluation repository (OpenHands/evaluation).

    Args:
        add_llm_config: Whether to add the llm_config_path positional argument.

    Returns:
        ArgumentParser instance with common benchmark arguments (no defaults).
    """
    parser = argparse.ArgumentParser(description="Run Evaluation inference")
    if add_llm_config:
        parser.add_argument(
            "llm_config_path",
            type=str,
            help="Path to JSON LLM configuration",
        )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
    )
    parser.add_argument("--split", type=str, help="Dataset split")
    parser.add_argument(
        "--workspace",
        type=str,
        default="remote",
        choices=["docker", "remote"],
        help="Type of workspace to use (default: remote)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="Maximum iterations (default: 500)",
    )
    parser.add_argument("--num-workers", type=int, help="Number of inference workers")
    parser.add_argument("--note", type=str, help="Optional evaluation note")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_outputs",
        help="Evaluation output directory",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=0,
        help="Limit number of instances to evaluate (0 = no limit)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        help="Maximum number of attempts for iterative mode (min: 1)",
    )

    # Add critic arguments (no default)
    add_critic_args(parser)

    parser.add_argument(
        "--select",
        type=str,
        help="Path to text file containing instance IDs to select (one per line)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Maximum retries for instances that throw exceptions",
    )
    return parser
