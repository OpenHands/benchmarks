"""Shared Harbor execution helpers for benchmark compatibility wrappers."""

from __future__ import annotations

import json
import os
import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from openhands.sdk import LLM, get_logger


logger = get_logger(__name__)


class HarborCredentialMode(str, Enum):
    """How LLM credentials are forwarded to Harbor/OpenHands SDK."""

    AGENT_ENV_FLAGS = "agent_env_flags"
    PROCESS_ENV = "process_env"


def check_harbor_installed(
    harbor_executable: str = "harbor",
    probe_arg: str = "--help",
) -> bool:
    """Return whether the Harbor CLI is installed and responds successfully."""
    try:
        result = subprocess.run(
            [harbor_executable, probe_arg],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _probe_harbor_run_help(harbor_executable: str) -> str:
    """Run harbor run --help and return combined stdout+stderr, or empty string if not found."""
    try:
        result = subprocess.run(
            [harbor_executable, "run", "--help"],
            capture_output=True,
            text=True,
        )
        return f"{result.stdout}\n{result.stderr}"
    except FileNotFoundError:
        return ""


def get_supported_task_filter_flag(harbor_executable: str) -> str:
    """Detect whether Harbor expects --task-name or --include-task-name."""
    help_text = _probe_harbor_run_help(harbor_executable)
    supported_flags = set(re.findall(r"(?<![\w-])--[a-z0-9-]+", help_text))
    if "--include-task-name" in supported_flags:
        return "--include-task-name"
    if "--task-name" in supported_flags:
        return "--task-name"
    return "--include-task-name"


def get_supported_agent_name(
    harbor_executable: str,
    default_agent_name: str = "openhands-sdk",
) -> str:
    """Detect whether Harbor exposes the OpenHands agent as openhands or openhands-sdk."""
    help_text = _probe_harbor_run_help(harbor_executable)
    compact_help_text = re.sub(r"[^a-z0-9-]+", "", help_text.lower())
    if "openhands-sdk" in compact_help_text:
        return "openhands-sdk"
    if "openhands" in compact_help_text:
        return "openhands"
    return default_agent_name


def _secret_value(value: object) -> str:
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()  # type: ignore[no-any-return, attr-defined]
    return str(value)


def run_harbor_evaluation(
    llm: LLM,
    dataset: str,
    output_dir: str,
    *,
    harbor_executable: str = "harbor",
    agent_name: str = "openhands-sdk",
    dataset_is_path: bool = False,
    num_workers: int = 1,
    task_ids: list[str] | None = None,
    n_limit: int | None = None,
    task_filter_flag: str = "--task-name",
    normalize_task_id: Callable[[str], str] | None = None,
    credential_mode: HarborCredentialMode = HarborCredentialMode.AGENT_ENV_FLAGS,
    retry_legacy_task_flag: bool = False,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> Path:
    """Run Harbor and return the directory containing Harbor job outputs.

    The ``subprocess_run`` parameter is a testing seam; pass a fake callable
    in tests rather than patching the subprocess module.
    """
    harbor_output_dir = Path(output_dir) / "harbor_output"
    harbor_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        harbor_executable,
        "run",
        "--path" if dataset_is_path else "-d",
        dataset,
        "-a",
        agent_name,
        "-m",
        llm.model,
        "--jobs-dir",
        str(harbor_output_dir.resolve()),
        "--n-concurrent",
        str(num_workers),
    ]

    env: dict[str, str] | None = None
    if credential_mode == HarborCredentialMode.AGENT_ENV_FLAGS:
        if llm.api_key:
            cmd.extend(["--ae", f"LLM_API_KEY={_secret_value(llm.api_key)}"])
        if llm.base_url:
            cmd.extend(["--ae", f"LLM_BASE_URL={llm.base_url}"])
    elif credential_mode == HarborCredentialMode.PROCESS_ENV:
        env = os.environ.copy()
        if llm.api_key:
            env["LLM_API_KEY"] = _secret_value(llm.api_key)
        if llm.base_url:
            env["LLM_BASE_URL"] = llm.base_url

    if task_ids:
        normalize = normalize_task_id or (lambda task_id: task_id)
        for task_id in task_ids:
            cmd.extend([task_filter_flag, normalize(task_id)])

    if n_limit is not None:
        cmd.extend(["--n-tasks", str(n_limit)])

    safe_cmd = [
        "***" if prev == "--ae" and part.startswith("LLM_") else part
        for prev, part in zip([""] + cmd, cmd)
    ]
    logger.info(f"Running harbor command: {' '.join(safe_cmd)}")
    logger.info(f"Output directory: {harbor_output_dir}")

    try:
        result = subprocess_run(cmd, capture_output=True, text=True, env=env)

        if (
            result.returncode != 0
            and retry_legacy_task_flag
            and task_ids
            and task_filter_flag == "--task-name"
            and "No such option: --task-name" in result.stderr
        ):
            fallback_cmd = [
                "--include-task-name" if part == "--task-name" else part for part in cmd
            ]
            logger.warning(
                "Harbor does not support --task-name; retrying with --include-task-name"
            )
            result = subprocess_run(
                fallback_cmd, capture_output=True, text=True, env=env
            )

        if result.returncode != 0:
            logger.error(f"Harbor command failed with code {result.returncode}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            raise RuntimeError(f"Harbor evaluation failed: {result.stderr}")

        logger.info("Harbor evaluation completed successfully")
        logger.info(f"stdout: {result.stdout}")
    except FileNotFoundError:
        raise RuntimeError(
            "Harbor CLI not found. Please install harbor: pip install harbor"
        )

    return harbor_output_dir


def find_job_dir(harbor_output_dir: Path) -> Path:
    """Find the latest Harbor job directory inside an output directory."""
    candidates = [
        d
        for d in harbor_output_dir.iterdir()
        if d.is_dir() and (d / "result.json").exists()
    ]
    if not candidates:
        raise RuntimeError(
            f"No harbor job directory found in {harbor_output_dir}. "
            f"Expected a timestamp-named directory containing result.json."
        )
    return sorted(candidates)[-1]


def convert_harbor_to_eval_output(
    harbor_output_dir: Path,
    eval_output_path: Path,
    *,
    canonicalize_instance_id: Callable[[str], str] | None = None,
) -> None:
    """Convert Harbor trial results to OpenHands benchmark output.jsonl format."""
    logger.info(f"Converting harbor output from {harbor_output_dir}")

    canonicalize = canonicalize_instance_id or (lambda instance_id: instance_id)
    job_dir = find_job_dir(harbor_output_dir)
    logger.info(f"Using harbor job directory: {job_dir}")

    result_files = [f for f in job_dir.glob("*/result.json") if f.parent != job_dir]
    if not result_files:
        raise RuntimeError(
            f"No trial result files found in {job_dir}. "
            f"Expected result.json files in trial subdirectories."
        )

    logger.info(f"Found {len(result_files)} trial results in {job_dir}")

    results: list[dict] = []
    errors: list[dict] = []

    for result_file in result_files:
        try:
            with open(result_file) as f:
                trial = json.load(f)

            instance_id = canonicalize(trial.get("task_name", result_file.parent.name))

            if trial.get("exception_info"):
                errors.append(
                    {
                        "instance_id": instance_id,
                        "error": str(trial["exception_info"]),
                        "test_result": {},
                    }
                )
                continue

            verifier_result = trial.get("verifier_result", {})
            rewards = verifier_result.get("rewards", {})
            passed = rewards.get("reward", 0.0) > 0
            agent_result = trial.get("agent_result", {})

            eval_entry = {
                "instance_id": instance_id,
                "test_result": {
                    "trial_name": trial.get("trial_name"),
                    "trial_uri": trial.get("trial_uri"),
                    "rewards": rewards,
                    "passed": passed,
                },
                "instruction": "",
                "error": None,
                "history": [],
                "metrics": {
                    "total_prompt_tokens": agent_result.get("n_input_tokens") or 0,
                    "total_completion_tokens": (
                        agent_result.get("n_output_tokens") or 0
                    ),
                    "total_cost_usd": agent_result.get("cost_usd") or 0.0,
                },
            }
            results.append(eval_entry)
            logger.info(
                f"Processed trial {instance_id}: reward={rewards.get('reward', 'N/A')}"
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to process result file {result_file}: {e}")
            errors.append(
                {
                    "instance_id": canonicalize(result_file.parent.name),
                    "error": str(e),
                    "test_result": {},
                }
            )

    if not results and not errors:
        raise RuntimeError(f"No trials processed from {harbor_output_dir}")

    if not results:
        logger.warning(
            f"All {len(errors)} trials failed in {harbor_output_dir}; "
            "writing error entries for downstream reporting"
        )

    with open(eval_output_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
        for entry in errors:
            f.write(json.dumps(entry) + "\n")

    logger.info(
        f"Wrote {len(results)} successful + {len(errors)} failed entries "
        f"to {eval_output_path}"
    )
