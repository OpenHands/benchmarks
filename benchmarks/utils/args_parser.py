"""
Argument parsing utilities for SWE-bench benchmarks.
"""

import argparse


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
        default="princeton-nlp/SWE-bench_Verified",
        help="Dataset name",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--max-iterations", type=int, default=100, help="Maximum iterations"
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of evaluation workers"
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
        default=None,
        help="Limit number of instances to evaluate",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum number of attempts for iterative mode (default: 3, min: 1)",
    )
    parser.add_argument(
        "--critic",
        type=str,
        required=True,
        help="Critic to use for evaluating instance success",
    )

    # Critic is now required

    # No custom validation needed
    return parser
