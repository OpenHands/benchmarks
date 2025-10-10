from __future__ import annotations

import json
import os
import threading

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from benchmarks.utils.models import EvalMetadata
from openhands.sdk import (
    LLM,
    get_logger,
    __version__
)


logger = get_logger(__name__)



def construct_eval_output_dir(
    base_dir: str,
    dataset_name: str,
    model_name: str,
    max_iterations: int,
    eval_note: str
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


def read_completed_instances(output_file: str) -> set:
    """Read completed instance IDs from existing output file."""
    completed_instances = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            result = json.loads(line)
                            if "instance_id" in result:
                                completed_instances.add(result["instance_id"])
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning(f"Error reading existing results from {output_file}: {e}")
    return completed_instances
