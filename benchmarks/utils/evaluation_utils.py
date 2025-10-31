from __future__ import annotations

import fcntl
import os
from typing import Callable

from benchmarks.utils.models import EvalInstance, EvalOutput
from openhands.sdk import get_logger


logger = get_logger(__name__)


def construct_eval_output_dir(
    base_dir: str,
    dataset_name: str,
    model_name: str,
    max_iterations: int,
    eval_note: str,
) -> str:
    """Construct the structured evaluation output directory path."""
    # Format: eval_out/<dataset>-<split>/<agent_config>/
    # <llm>_maxiter_<maxiter>_N_<version>-<hint>-<exp_name>-run_<run_number>

    # Create LLM config string
    folder = f"{model_name}_maxiter_{max_iterations}"
    if eval_note:
        folder += f"_N_{eval_note}"

    # Construct full path
    eval_output_dir = os.path.join(base_dir, dataset_name, folder)
    os.makedirs(eval_output_dir, exist_ok=True)

    return eval_output_dir


def get_default_on_result_writer(
    output_path: str,
) -> Callable[[EvalInstance, EvalOutput], None]:
    """
    Create a default callback that writes successful evaluation results to a JSONL file.

    Exceptions (outputs with error field) are not written to the file.

    Args:
        output_path: Path to the output JSONL file

    Returns:
        A callback function that can be passed to evaluator.run(on_result=...)
    """

    def _cb(instance: EvalInstance, out: EvalOutput) -> None:
        if out.error:  # Skip writing if there's an error
            return
        with open(output_path, "a") as f:
            # Use exclusive lock to prevent race
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(out.model_dump_json() + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

    return _cb
