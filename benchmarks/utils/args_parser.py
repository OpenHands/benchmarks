"""
Argument parsing utilities for SWE-bench benchmarks.
"""

import argparse


def get_parser():
    """Create and return argument parser.

    Returns:
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description="Run Evaluation inference")
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Verified",
        help="Dataset name",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--llm-config-path",
        type=str,
        required=True,
        help="Path to JSON LLM configuration",
    )
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
        default="./eval_out",
        help="Evaluation output directory",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=-1,
        help="Limit number of instances to evaluate",
    )
    return parser
