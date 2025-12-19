from __future__ import annotations

import os
from pathlib import Path

from benchmarks.utils.constants import ARTIFACTS_DIRNAME
from benchmarks.utils.output_schema import load_output_file
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

    eval_output_dir = Path(base_dir) / dataset_name / folder
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    return str(eval_output_dir)


def get_attempt_artifact_dir(
    eval_output_dir: str, instance_id: str, attempt: int
) -> Path:
    """Return the standardized artifact directory for an attempt."""
    artifacts_dir = (
        Path(eval_output_dir)
        / ARTIFACTS_DIRNAME
        / instance_id
        / f"attempt_{attempt}"
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "logs").mkdir(exist_ok=True)
    return artifacts_dir


def generate_error_logs_summary(eval_output_dir: str) -> None:
    """
    Generate an ERROR_LOGS.txt file that lists all failed instances and their log locations.

    This makes it easy to quickly find and navigate to error logs in the GCS artifact folder.

    Args:
        eval_output_dir: Path to the evaluation output directory
    """
    output_file = Path(eval_output_dir) / "output.jsonl"
    if not output_file.exists():
        logger.info("No output.jsonl found, skipping ERROR_LOGS.txt generation")
        return

    try:
        outputs = load_output_file(output_file)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to read %s: %s", output_file, exc)
        return

    error_instances = [
        out
        for out in outputs
        if out.status == "error" or (out.resolved is False and out.status != "skipped")
    ]

    if not error_instances:
        logger.info("No error instances found, skipping ERROR_LOGS.txt generation")
        return

    summary_path = Path(eval_output_dir) / "ERROR_LOGS.txt"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("FAILED INSTANCES - QUICK REFERENCE\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total failed instances: {len(error_instances)}\n\n")

            for i, error in enumerate(error_instances, 1):
                instance_id = error.instance_id
                error_msg = error.error or "No error message"
                attempt = error.attempt

                artifact_path = Path(error.artifacts_url)
                if not artifact_path.is_absolute():
                    artifact_path = Path(eval_output_dir) / artifact_path
                logs_dir = artifact_path / "logs"

                f.write(f"[{i}] Instance ID: {instance_id} (attempt {attempt})\n")
                f.write(f"    Error: {error_msg}\n")
                f.write(f"    Main log: {logs_dir / 'instance.log'}\n")
                f.write(f"    Output log: {logs_dir / 'instance.output.log'}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("NAVIGATION TIPS:\n")
            f.write("=" * 80 + "\n")
            f.write("1. Download the full results archive from GCS\n")
            f.write("2. Extract the tar.gz file\n")
            f.write("3. Navigate to logs/ directory under artifacts/\n")
            f.write("4. Open the log files listed above for detailed error information\n")
            f.write("\n")
            f.write("Main logs contain evaluation framework messages.\n")
            f.write("Output logs contain agent conversation output.\n")

        logger.info(
            "Generated ERROR_LOGS.txt with %d failed instances at %s",
            len(error_instances),
            summary_path,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to generate ERROR_LOGS.txt: {e}")
