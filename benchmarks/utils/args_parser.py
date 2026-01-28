"""
Argument parsing utilities for benchmarks.

Default values are aligned with the evaluation repository (OpenHands/evaluation)
to ensure consistency between local development and production runs.

Benchmark-specific values should be set via parser.set_defaults() in each
benchmark's run_infer.py to override these common defaults.
"""

import argparse

from benchmarks.utils.critics import add_critic_args


def get_parser(add_llm_config: bool = True) -> argparse.ArgumentParser:
    """Create and return argument parser with common defaults.

    Default values match the most common settings used across benchmarks
    in the evaluation repository. Individual benchmarks can override
    these using parser.set_defaults() before calling parse_args().

    Args:
        add_llm_config: Whether to add the llm_config_path positional argument.

    Returns:
        ArgumentParser instance with common benchmark arguments.
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
        help="Dataset name (each benchmark sets its default via set_defaults)",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--workspace",
        type=str,
        default="remote",
        choices=["docker", "remote"],
        help="Type of workspace to use (default: remote)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=500, help="Maximum iterations"
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of inference workers"
    )
    parser.add_argument("--note", type=str, default="initial", help="Evaluation note")
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
        help="Limit number of instances to evaluate",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum number of attempts for iterative mode (default: 3, min: 1)",
    )

    # Add critic arguments (default: finish_with_patch)
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
        default=3,
        help="Maximum retries for instances that throw exceptions (default: 3)",
    )
    return parser
