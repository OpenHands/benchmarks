from __future__ import annotations

import json
import os
import threading

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from benchmarks.utils.shared import EvalMetadata
from openhands.sdk import (
    LLM,
    get_logger,
)


# from openhands.tools import (
#     BashTool,
#     FileEditorTool,
# )


logger = get_logger(__name__)


def get_instruction(
    instance: pd.Series, metadata: EvalMetadata, workspace_path: str, prompt_path: str
) -> str:
    """Generate instruction for the agent."""
    workspace_dir_name = instance.repo.split("/")[-1]
    assert metadata.details is not None

    # Set up Jinja2 environment
    prompts_dir = os.path.dirname(prompt_path)
    template_name = os.path.basename(prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        "instance": instance,
        "workspace_dir_name": workspace_dir_name,
        "actual_workspace_path": workspace_path,
        "metadata": metadata,
    }

    context["test_instructions"] = ""

    # Render the instruction
    instruction = template.render(context)
    return instruction


def make_metadata(
    llm: LLM,
    dataset_name,
    max_iterations,
    eval_output_dir,
    details=None,
    dataset=None,
    data_split=None,
    prompt_path=None,
    eval_n_limit=None,
    env_setup_commands=None,
):
    """Create evaluation metadata."""
    return EvalMetadata(
        llm=llm,
        data_split=data_split or dataset_name,
        max_iterations=max_iterations,
        eval_output_dir=eval_output_dir,
        details=details,
        dataset=dataset,
        prompt_path=prompt_path,
        eval_n_limit=eval_n_limit,
        env_setup_commands=env_setup_commands,
    )


def construct_eval_output_dir(base_dir, dataset_name, model, max_iterations, eval_note):
    """Construct the structured evaluation output directory path."""
    # Format: eval_out/<dataset>-<split>/<agent_config>/
    # <llm>_maxiter_<maxiter>_N_<version>-<hint>-<exp_name>-run_<run_number>

    # Create LLM config string
    llm_config_str = f"{model}_maxiter_{max_iterations}"

    # Add version and note information
    version = "v1"  # Default version
    hint_status = "no-hint"  # Default hint status

    if eval_note:
        llm_config_str += f"_N_{version}-{hint_status}-{eval_note}-run_1"
    else:
        llm_config_str += f"_N_{version}-{hint_status}-run_1"

    # Construct full path
    eval_output_dir = os.path.join(base_dir, dataset_name, llm_config_str)
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


def write_output_to_file(instance, process_instance, result, output_file):
    # Save result using the complete format
    result_dict = result.model_dump(mode="json")

    logger.info(f"Writing result for {instance.instance_id} to {output_file}")
    logger.info(f"Result dict keys: {list(result_dict.keys())}")
    git_patch_len = len(result_dict.get("test_result", {}).get("git_patch", ""))
    logger.info(f"Result dict git_patch length: {git_patch_len}")
    logger.info(f"Result dict history length: {len(result_dict.get('history', []))}")

    # Write to output file (thread-safe)
    import json

    # Use a lock to ensure thread-safe file writing
    if not hasattr(process_instance, "_file_lock"):
        process_instance._file_lock = threading.Lock()

    with process_instance._file_lock:
        with open(output_file, "a") as f:
            json_line = json.dumps(result_dict) + "\n"
            f.write(json_line)
            f.flush()  # Ensure it's written immediately
            logger.info(
                f"Successfully wrote {len(json_line)} characters to output file"
            )
