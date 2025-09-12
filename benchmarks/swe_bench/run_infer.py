from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from typing import Literal

import modal
import pandas as pd
from datasets import load_dataset, Dataset
from jinja2 import Environment, FileSystemLoader
from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    ImageContent,
    Message,
    TextContent,
    Tool,
    get_logger,
)
from openhands.tools import (
    BashTool,
    FileEditorTool,
)

from benchmarks.swe_bench.binary_patch_utils import (
    remove_binary_diffs,
)
from benchmarks.swe_bench.resource.swt_bench_constants import (
    MAP_REPO_TO_TEST_FRAMEWORK_VERBOSE,
)
from benchmarks.utils.shared import EvalMetadata, EvalOutput, EvalException


logger = get_logger(__name__)
BenchMode = Literal["swe", "swt", "swt-ci"]

# Global variable to track dataset type
DATASET_TYPE = "SWE-bench"


def set_dataset_type(dataset_name: str):
    """Set dataset type based on dataset name."""
    global DATASET_TYPE
    name_lower = dataset_name.lower()

    if "swe-gym" in name_lower:
        DATASET_TYPE = "SWE-Gym"
    elif "swe-bench-live" in name_lower:
        DATASET_TYPE = "SWE-bench-Live"
    elif "multimodal" in name_lower:
        DATASET_TYPE = "Multimodal"
    else:
        DATASET_TYPE = "SWE-bench"

    logger.info(f"Dataset type set to: {DATASET_TYPE}")


def _get_workspace_dir_name(instance: pd.Series) -> str:
    """Extract repo name from instance.repo (e.g., "django/django" -> "django")"""
    repo_name = instance.repo.split("/")[-1]
    return repo_name

def get_instruction(
    instance: pd.Series, metadata: EvalMetadata, workspace_path: str
) -> str:
    """Generate instruction for the agent."""
    workspace_dir_name = _get_workspace_dir_name(instance)
    assert metadata.details is not None
    mode = metadata.details["mode"]
    llm_model = metadata.llm_config.model

    # Determine the template file based on mode and LLM
    if mode.startswith("swt"):
        template_name = "swt.j2"
    elif mode == "swe":
        if "claude" in llm_model:
            template_name = "swe_default.j2"
        elif "gpt-4.1" in llm_model:
            template_name = "swe_gpt4.j2"
        else:
            template_name = "swe_default.j2"  # Default for 'swe' mode
    else:
        logger.error(f"Unexpected evaluation mode: {mode}. Falling back to default.")
        template_name = "swe_default.j2"

    # Set up Jinja2 environment
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        "instance": instance,
        "workspace_dir_name": workspace_dir_name,
        "actual_workspace_path": workspace_path,
        "metadata": metadata,
    }

    # Add specific context for swt-ci mode if needed
    if mode == "swt-ci":
        context["test_instructions"] = (
            f"The following command can be used to run the tests: `{list(MAP_REPO_TO_TEST_FRAMEWORK_VERBOSE[instance.repo].values())[0]}`. Make sure they fail in the expected way.\n"
        )
    else:
        context["test_instructions"] = ""

    # Render the instruction
    instruction = template.render(context)
    return instruction




def setup_workspace(instance: pd.Series, workspace_root: str) -> str:
    """Setup workspace for the instance by cloning the repository."""
    repo_name = instance.repo  # e.g., "django/django"
    base_commit = instance.base_commit
    workspace_dir_name = _get_workspace_dir_name(instance)
    workspace_path = os.path.join(workspace_root, workspace_dir_name)

    # Construct GitHub URL
    repo_url = f"https://github.com/{repo_name}.git"

    logger.info(f"Setting up workspace for {repo_name} at {workspace_path}")

    # Remove existing directory if it exists
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path)

    # Clone the repository
    try:
        subprocess.run(
            ["git", "clone", repo_url, workspace_path],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Successfully cloned {repo_url} to {workspace_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository {repo_url}: {e.stderr}")
        raise EvalException(f"Failed to clone repository: {e.stderr}")

    # Checkout the base commit
    try:
        subprocess.run(
            ["git", "checkout", base_commit],
            cwd=workspace_path,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Successfully checked out base commit {base_commit}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to checkout base commit {base_commit}: {e.stderr}")
        raise EvalException(f"Failed to checkout base commit: {e.stderr}")

    return workspace_path


def initialize_workspace(workspace_path: str, instance: pd.Series):
    """Initialize the workspace with necessary setup."""
    logger.info("-" * 30)
    logger.info("BEGIN Workspace Initialization")
    logger.info("-" * 30)

    # Set up environment variables and git configuration
    env_setup_commands = [
        f"export SWE_INSTANCE_ID={instance['instance_id']}",
        "export PIP_CACHE_DIR=~/.cache/pip",
        'git config --global core.pager ""',
        "git config --global diff.binary false",
    ]

    for cmd in env_setup_commands:
        try:
            subprocess.run(
                cmd,
                shell=True,
                cwd=workspace_path,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Successfully executed: {cmd}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to execute {cmd}: {e.stderr}")
            raise EvalException(f"Failed to initialize workspace: {e.stderr}")

    # Create necessary directories
    swe_util_dir = "/swe_util/eval_data/instances"
    os.makedirs(swe_util_dir, exist_ok=True)

    # Write instance data
    swe_instance_json_name = "swe-bench-instance.json"
    instance_file_path = os.path.join(swe_util_dir, swe_instance_json_name)
    with open(instance_file_path, "w") as f:
        if not isinstance(instance, dict):
            json.dump([instance.to_dict()], f)
        else:
            json.dump([instance], f)

    # Copy setup scripts
    script_dir = os.path.dirname(__file__)
    if DATASET_TYPE == "SWE-bench-Live":
        entry_script_path = "instance_swe_entry_live.sh"
    else:
        entry_script_path = "instance_swe_entry.sh"

    src_script = os.path.join(script_dir, f"scripts/setup/{entry_script_path}")
    dst_script = f"/swe_util/{entry_script_path}"
    if os.path.exists(src_script):
        shutil.copy2(src_script, dst_script)

        # Execute the setup script
        try:
            subprocess.run(
                f"source {dst_script}",
                shell=True,
                cwd=workspace_path,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Successfully executed setup script: {entry_script_path}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Setup script execution failed (non-fatal): {e.stderr}")


def get_git_patch(workspace_path: str) -> str:
    """Get git patch from the workspace."""
    logger.info("-" * 30)
    logger.info("BEGIN Git Patch Extraction")
    logger.info("-" * 30)

    try:
        # Change to workspace directory
        os.chdir(workspace_path)

        # Configure git
        subprocess.run(
            ["git", "config", "--global", "core.pager", '""'],
            check=True,
            capture_output=True,
            text=True,
        )

        # Remove any nested git repositories
        result = subprocess.run(
            ["find", ".", "-type", "d", "-name", ".git", "-not", "-path", "./.git"],
            capture_output=True,
            text=True,
        )
        git_dirs = [p for p in result.stdout.strip().split("\n") if p]
        for git_dir in git_dirs:
            shutil.rmtree(git_dir)
            logger.info(f"Removed nested git directory: {git_dir}")

        # Check if this is a git repository
        try:
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            logger.error("Current directory is not a git repository")
            return ""

        # Add all changes
        subprocess.run(["git", "add", "-A"], check=True, capture_output=True, text=True)

        # Get the diff
        result = subprocess.run(
            ["git", "diff", "--cached"], capture_output=True, text=True
        )
        git_patch = result.stdout

        # Remove binary diffs if present
        git_patch = remove_binary_diffs(git_patch)

        logger.info(f"Generated git patch with {len(git_patch)} characters")
        return git_patch

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate git patch: {e.stderr}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error generating git patch: {str(e)}")
        return ""


app = modal.App(name="swe-bench-eval")

image = (
    modal.Image.debian_slim()
    .apt_install(
        "git",
        "curl",
        "wget",
        "unzip",
        "python3-pip",
        "build-essential",
        "libssl-dev",
        "libffi-dev",
        "python3-dev",
    )
    .pip_install(
        "git+openhands-sdk @ git+https://github.com/All-Hands-AI/agent-sdk.git#subdirectory=openhands/sdk",
        "openhands-tools @ git+https://github.com/All-Hands-AI/agent-sdk.git#subdirectory=openhands/tools"
    )
)

@app.function()
def process_instance_simplified(
    instance: pd.Series,
    instruction: str,
    metadata: EvalMetadata
) -> EvalOutput:
    """Process a single instance using the simplified SDK approach."""
    logger.info(f"Starting evaluation for instance {instance.instance_id}")

    pathlib.Path("/workspace").mkdir(parents=True, exist_ok=True)
    workspace_path = setup_workspace(instance, "/workspace")
    initialize_workspace(workspace_path, instance)

    llm = metadata.llm

    # Setup tools with the workspace
    tools = [
        BashTool.create(working_dir=workspace_path),
        FileEditorTool.create(),
    ]

    # Create agent
    agent = Agent(llm=llm, tools=tools)

    # Create conversation with callback
    conversation = Conversation(agent=agent)


    # Handle multimodal content if present
    if "image_assets" in instance:
        assets = json.loads(instance["image_assets"])
        assert "problem_statement" in assets, (
            "problem_statement is required in image_assets"
        )
        image_urls = assets["problem_statement"]
        message = Message(
            role="user",
            content=[
                TextContent(text=instruction),
                # TODO: will fix this in next version of SDK
                # ImageContent(image_urls=image_urls),
            ],
        )
    else:
        message = Message(role="user", content=[TextContent(text=instruction)])

    # Send message and run conversation
    conversation.send_message(message)
    conversation.run()

    history = list(conversation.state.events)

    logger.info(
        f"Conversation completed with {len(history)} events"
    )

    # Get git patch
    git_patch = get_git_patch(workspace_path)


    logger.info(f"Completed evaluation for instance {instance.instance_id}")
    logger.info(f"Git patch length: {len(git_patch)} characters")

    return EvalOutput(
        instance_id=instance.instance_id,
        test_result={
            'git_patch': git_patch,
        },
        instruction=instruction,
        metadata=EvalMetadata(
            llm=metadata.llm,
            max_iterations=metadata.max_iterations,
            eval_output_dir=metadata.eval_output_dir,
            dataset=metadata.dataset,
            data_split=metadata.data_split,
            details=metadata.details,
        ),
        history=history,
    )

def get_evaluation_parser():
    """Get argument parser for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--max-iterations",
        dest="max_iterations",
        type=int,
        default=int(os.environ.get("MAX_ITERATIONS", "50")),
    )
    parser.add_argument(
        "--eval-output-dir",
        dest="eval_output_dir",
        type=str,
        default=os.environ.get("EVAL_OUTPUT_DIR", "./eval_out"),
    )
    parser.add_argument(
        "--eval-num-workers",
        dest="eval_num_workers",
        type=int,
        default=int(os.environ.get("EVAL_NUM_WORKERS", "1")),
    )
    parser.add_argument(
        "--eval-n-limit",
        dest="eval_n_limit",
        type=int,
        default=int(os.environ.get("EVAL_N_LIMIT", "0")),
    )
    parser.add_argument(
        "--eval-note",
        dest="eval_note",
        type=str,
        default=os.environ.get("EVAL_NOTE", ""),
    )
    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        default=os.environ.get("MODEL", "claude-3-5-sonnet-latest"),
    )
    parser.add_argument(
        "--llm-config",
        dest="llm_config",
        type=str,
        default=os.environ.get("LLM_CONFIG", None),
    )
    return parser


def filter_dataset(dataset: pd.DataFrame, filter_column: str) -> pd.DataFrame:
    """Filter dataset based on environment variables."""
    # This is a simplified version - you may need to add more filtering logic
    return dataset


def make_metadata(
    llm: LLM,
    dataset_name,
    max_iterations,
    eval_output_dir,
    details=None,
):
    """Create evaluation metadata."""
    return EvalMetadata(
        llm=llm,
        data_split=dataset_name,
        max_iterations=max_iterations,
        eval_output_dir=eval_output_dir,
        details=details,
    )


def construct_eval_output_dir(
    base_dir,
    dataset_name,
    model,
    max_iterations,
    eval_note
):
    """Construct the structured evaluation output directory path."""
    # Format: eval_out/<dataset>-<split>/<agent_config>/<llm>_maxiter_<maxiter>_N_<version>-<hint>-<exp_name>-run_<run_number>

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

    return eval_output_dir


def prepare_dataset(
    dataset: pd.DataFrame, output_file: str, n_limit: int
) -> pd.DataFrame:
    """Prepare dataset for evaluation."""
    if n_limit > 0:
        dataset = dataset.head(n_limit)
    return dataset


def run_evaluation_simplified(
    instances: pd.DataFrame, metadata: EvalMetadata, output_file: str
):
    """Run evaluation on instances."""
    output_dir = os.path.dirname(output_file)
    if output_dir:  # Only create directory if dirname is not empty
        os.makedirs(output_dir, exist_ok=True)

    # Create empty output file
    with open(output_file, "w") as f:
        pass

    results = []
    for idx, instance in instances.iterrows():
        logger.info(f"Processing instance {instance.instance_id}")
        # Get instruction
        workspace_path = os.path.join("/workspace", _get_workspace_dir_name(instance))
        instruction = get_instruction(instance, metadata, workspace_path)
        result = process_instance_simplified.remote(instance, instruction, metadata)

        # Save result using the complete format
        result_dict = result.model_dump()
        if result.error:
            result_dict["error"] = result.error
        results.append(result_dict)

        logger.info(f"Writing result for {instance.instance_id} to {output_file}")
        logger.info(f"Result dict keys: {list(result_dict.keys())}")
        git_patch_len = len(result_dict.get("test_result", {}).get("git_patch", ""))
        logger.info(f"Git patch length: {git_patch_len}")

        # Write to output file
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "a") as f:
            json_line = json.dumps(result_dict) + "\n"
            f.write(json_line)
            f.flush()  # Ensure it's written immediately
            logger.info(
                f"Successfully wrote {len(json_line)} characters to output file"
            )

def main():
    DATASET = "princeton-nlp/SWE-bench"
    SPLIT = "test"
    MODE = "swe"  # "swe", "swt", or "swt-ci"
    MODEL = "litellm-proxy/anthropic/claude-sonnet-4-20250514"
    EVAL_OUTPUT_DIR = "./eval_out"
    MAX_ITERATIONS = 100
    EVAL_N_LIMIT = 1
    EVAL_NOTE = "initial"

    # Load dataset
    dataset = load_dataset(DATASET, split=SPLIT)
    set_dataset_type(DATASET)
    assert isinstance(dataset, Dataset)
    _df = dataset.to_pandas()
    assert isinstance(_df, pd.DataFrame)

    swe_bench_tests = filter_dataset(_df, "instance_id")
    logger.info(
        f"Loaded dataset {DATASET} with split {SPLIT}: {len(swe_bench_tests)} tasks"
    )

    # Create LLM instance
    api_key = os.getenv("LITELLM_API_KEY")
    if not api_key:
        raise ValueError("LITELLM_API_KEY environment variable is not set")
    llm = LLM(
        model=MODEL,
        api_key=SecretStr(api_key),
        base_url="https://llm-proxy.eval.all-hands.dev",
        temperature=0,
    )

    details = {"mode": MODE}
    dataset_description = (
        DATASET.replace("/", "__") + "-" + SPLIT.replace("/", "__")
    )

    # Construct proper structured output directory path
    structured_output_dir = construct_eval_output_dir(
        base_dir=EVAL_OUTPUT_DIR,
        dataset_name=dataset_description,
        model=llm.model,
        max_iterations=MAX_ITERATIONS,
        eval_note=EVAL_NOTE,
    )

    metadata = make_metadata(
        llm,
        dataset_description,
        MAX_ITERATIONS,
        structured_output_dir,
        details=details,
    )

    # Create output directory and file
    os.makedirs(metadata.eval_output_dir, exist_ok=True)
    output_file = os.path.join(metadata.eval_output_dir, "output.jsonl")
    print(f"### OUTPUT FILE: {output_file} ###")

    # Prepare dataset
    instances = prepare_dataset(swe_bench_tests, output_file, EVAL_N_LIMIT)

    # Run evaluation
    run_evaluation_simplified(instances, metadata, output_file)

    logger.info("Evaluation completed!")
