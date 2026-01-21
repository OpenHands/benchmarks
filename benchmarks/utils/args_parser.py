"""
Argument parsing utilities for SWE-bench benchmarks.
"""

import argparse

from benchmarks.utils.constants import (
    DEFAULT_DATASET,
    DEFAULT_EVAL_LIMIT,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT,
    DEFAULT_WORKSPACE_TYPE,
)
from benchmarks.utils.critics import add_critic_args


def get_parser(add_llm_config: bool = True) -> argparse.ArgumentParser:
    """Create and return argument parser.

    Returns:
        ArgumentParser instance
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
        default=DEFAULT_DATASET,
        help="Dataset name",
    )
    parser.add_argument(
        "--split", type=str, default=DEFAULT_SPLIT, help="Dataset split"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=DEFAULT_WORKSPACE_TYPE,
        choices=["docker", "remote"],
        help=f"Type of workspace to use (default: {DEFAULT_WORKSPACE_TYPE})",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Maximum iterations",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of evaluation workers",
    )
    parser.add_argument("--note", type=str, default="initial", help="Evaluation note")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Evaluation output directory",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=DEFAULT_EVAL_LIMIT,
        help="Limit number of instances to evaluate",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Maximum number of attempts for iterative mode (default: {DEFAULT_MAX_ATTEMPTS}, min: 1)",
    )

    # Add critic arguments
    add_critic_args(parser)

    parser.add_argument(
        "--select",
        type=str,
        default=None,
        help="Path to text file containing instance IDs to select (one per line)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retries for instances that throw exceptions (default: {DEFAULT_MAX_RETRIES})",
    )
    return parser
