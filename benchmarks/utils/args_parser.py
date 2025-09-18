"""
Argument parsing utilities for SWE-bench benchmarks.
"""

import argparse


def parse_args(default_prompt_path: str):
    """Parse command line arguments.

    Args:
        default_prompt_path: Default path to the prompt template file

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Run SWE-bench inference")
    parser.add_argument(
        "--agent-cls", type=str, default="CodeActAgent", help="Agent class to use"
    )
    parser.add_argument(
        "--llm-config", type=str, required=True, help="LLM configuration"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=100, help="Maximum iterations"
    )
    parser.add_argument(
        "--eval-num-workers", type=int, default=1, help="Number of evaluation workers"
    )
    parser.add_argument(
        "--eval-note", type=str, default="initial", help="Evaluation note"
    )
    parser.add_argument(
        "--dataset", type=str, default="princeton-nlp/SWE-bench", help="Dataset name"
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default="./eval_out",
        help="Evaluation output directory",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=default_prompt_path,
        help="Path to prompt template file",
    )
    parser.add_argument(
        "--eval-n-limit",
        type=int,
        default=1,
        help="Limit number of instances to evaluate",
    )
    return parser.parse_args()
