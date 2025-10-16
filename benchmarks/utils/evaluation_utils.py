from __future__ import annotations

import os
from typing import Any

from openhands.sdk import LLM, get_logger

from .models import EvalMetadata


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


def make_metadata(
    llm: LLM,
    dataset: str,
    max_iterations: int,
    eval_output_dir: str,
    details: dict[str, Any] | None = None,
    data_split: str | None = None,
    prompt_path: str | None = None,
    eval_n_limit: int | None = None,
    env_setup_commands: list[str] | None = None,
    max_attempts: int = 1,
    critic_name: str = "default_critic",
) -> EvalMetadata:
    """Create EvalMetadata instance with the provided parameters."""
    # Handle None values by using the model defaults
    kwargs = {
        "llm": llm,
        "dataset": dataset,
        "max_iterations": max_iterations,
        "eval_output_dir": eval_output_dir,
        "details": details,
        "prompt_path": prompt_path,
        "env_setup_commands": env_setup_commands,
        "max_attempts": max_attempts,
        "critic_name": critic_name,
    }

    # Only set these if they're not None, let the model use defaults otherwise
    if data_split is not None:
        kwargs["dataset_split"] = data_split
    if eval_n_limit is not None:
        kwargs["eval_limit"] = eval_n_limit

    return EvalMetadata(**kwargs)
