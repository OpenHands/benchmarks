from __future__ import annotations

import fcntl
import json
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
) -> str:
    """Construct the structured evaluation output directory path."""
    # Format: eval_out/<dataset>-<split>/<agent_config>/
    # <llm>_sdk_<sdk_short_sha>_maxiter_<maxiter>_N_<user_note>/

    # Create LLM config string
    folder = f"{model_name}_sdk_{SDK_SHORT_SHA}_maxiter_{max_iterations}"
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
    Create a default callback that writes evaluation results to JSONL files.

    Successful results are written to output.jsonl.
    Failed results (with error field) are written to output_errors.jsonl.

    Args:
        output_path: Path to the main output JSONL file

    Returns:
        A callback function that can be passed to evaluator.run(on_result=...)
    """
    # Derive error output path from main output path
    error_output_path = output_path.replace(".jsonl", "_errors.jsonl")

    def _cb(instance: EvalInstance, out: EvalOutput) -> None:
        # Choose the appropriate file based on whether there's an error
        target_path = error_output_path if out.error else output_path

        with open(target_path, "a") as f:
            # Use exclusive lock to prevent race conditions in parallel execution
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(out.model_dump_json() + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

    return _cb


def generate_error_logs_summary(eval_output_dir: str) -> None:
    """
    Generate an ERROR_LOGS.txt file that lists all failed instances and their log locations.

    This makes it easy to quickly find and navigate to error logs in the GCS artifact folder.

    Args:
        eval_output_dir: Path to the evaluation output directory
    """
    error_output_path = os.path.join(eval_output_dir, "output_errors.jsonl")

    # Check if there are any errors
    if not os.path.exists(error_output_path):
        logger.info("No error instances found, skipping ERROR_LOGS.txt generation")
        return

    # Load error instances
    error_instances = []
    try:
        with open(error_output_path, "r") as f:
            for line in f:
                if line.strip():
                    error_instances.append(json.loads(line))
    except Exception as e:
        logger.warning(f"Failed to read error instances: {e}")
        return

    if not error_instances:
        logger.info("No error instances found, skipping ERROR_LOGS.txt generation")
        return

    # Generate summary file
    summary_path = os.path.join(eval_output_dir, "ERROR_LOGS.txt")
    try:
        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("FAILED INSTANCES - QUICK REFERENCE\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total failed instances: {len(error_instances)}\n\n")

            for i, error in enumerate(error_instances, 1):
                instance_id = error.get("instance_id", "unknown")
                error_msg = error.get("error", "No error message")

                f.write(f"[{i}] Instance ID: {instance_id}\n")
                f.write(f"    Error: {error_msg}\n")
                f.write(f"    Main log: logs/instance_{instance_id}.log\n")
                f.write(f"    Output log: logs/instance_{instance_id}.output.log\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("NAVIGATION TIPS:\n")
            f.write("=" * 80 + "\n")
            f.write("1. Download the full results archive from GCS\n")
            f.write("2. Extract the tar.gz file\n")
            f.write("3. Navigate to logs/ directory\n")
            f.write(
                "4. Open the log files listed above for detailed error information\n"
            )
            f.write("\n")
            f.write("Main logs contain evaluation framework messages.\n")
            f.write("Output logs contain agent conversation output.\n")

        logger.info(
            f"Generated ERROR_LOGS.txt with {len(error_instances)} failed instances at {summary_path}"
        )
    except Exception as e:
        logger.warning(f"Failed to generate ERROR_LOGS.txt: {e}")
