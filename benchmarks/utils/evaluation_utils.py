from __future__ import annotations

import fcntl
import os
from typing import Callable

from benchmarks.utils.models import EvalInstance, EvalOutput
from benchmarks.utils.version import SDK_SHORT_SHA
from openhands.sdk import get_logger


logger = get_logger(__name__)


def construct_eval_output_dir(
    base_dir: str,
    dataset_name: str,
    model_name: str,
    max_iterations: int,
    eval_note: str,
    workflow_memory_mode: str = "none",
) -> str:
    """Construct the structured evaluation output directory path."""
    # Format: eval_out/<dataset>-<split>/<agent_config>/
    # <llm>_sdk_<sdk_short_sha>_maxiter_<maxiter>_wfmem_<mode>_N_<user_note>/

    # Create LLM config string
    folder = f"{model_name}_sdk_{SDK_SHORT_SHA}_maxiter_{max_iterations}"

    # Add workflow memory mode if not "none"
    if workflow_memory_mode != "none":
        folder += f"_wfmem_{workflow_memory_mode}"

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
