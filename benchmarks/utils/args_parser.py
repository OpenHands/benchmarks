"""
Argument parsing utilities for benchmarks.
"""

import argparse

from benchmarks.utils import constants
from benchmarks.utils.critics import add_critic_args


def get_parser(add_llm_config: bool = True) -> argparse.ArgumentParser:
    """Create and return argument parser.

    Note: --dataset has no default. Each benchmark should set its own default
    using parser.set_defaults(dataset=<benchmark_specific_constant>).

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
        default=None,
        help="Dataset name (required unless benchmark provides default)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=constants.DEFAULT_SPLIT,
        help=f"Dataset split (default: {constants.DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=constants.DEFAULT_WORKSPACE,
        choices=["docker", "remote"],
        help=f"Type of workspace to use (default: {constants.DEFAULT_WORKSPACE})",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=constants.DEFAULT_MAX_ITERATIONS,
        help=f"Maximum iterations (default: {constants.DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=constants.DEFAULT_NUM_EVAL_WORKERS,
        help=f"Number of evaluation workers (default: {constants.DEFAULT_NUM_EVAL_WORKERS})",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=constants.DEFAULT_NOTE,
        help=f"Evaluation note (default: {constants.DEFAULT_NOTE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=constants.DEFAULT_OUTPUT_DIR,
        help=f"Evaluation output directory (default: {constants.DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=constants.DEFAULT_N_LIMIT,
        help=f"Limit number of instances to evaluate (default: {constants.DEFAULT_N_LIMIT})",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=constants.DEFAULT_MAX_ATTEMPTS,
        help=f"Maximum number of attempts for iterative mode (default: {constants.DEFAULT_MAX_ATTEMPTS}, min: 1)",
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
        default=constants.DEFAULT_MAX_RETRIES,
        help=f"Maximum retries for instances that throw exceptions (default: {constants.DEFAULT_MAX_RETRIES})",
    )
    return parser
