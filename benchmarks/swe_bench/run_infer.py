from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from typing import Literal

import pandas as pd
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader
from pydantic import SecretStr


# Ensure OpenHands SDK is importable
_SDK_DIR = os.environ.get("OPENHANDS_SDK")
if not _SDK_DIR:
    raise RuntimeError(
        "OPENHANDS_SDK environment variable is not set. "
        "Please set it to the path of your OpenHands SDK directory. "
        "Example: export OPENHANDS_SDK=/path/to/agent-sdk"
    )
if _SDK_DIR not in sys.path:
    sys.path.insert(0, _SDK_DIR)

# Import SDK components directly
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

# Import SWE-bench specific components
from swe_bench.binary_patch_utils import (
    remove_binary_diffs,
)
from swe_bench.resource.swt_bench_constants import (
    MAP_REPO_TO_TEST_FRAMEWORK_VERBOSE,
)


logger = get_logger("swe_bench_eval")

USE_HINT_TEXT = os.environ.get("USE_HINT_TEXT", "false").lower() == "true"
RUN_WITH_BROWSING = os.environ.get("RUN_WITH_BROWSING", "false").lower() == "true"
ENABLE_LLM_EDITOR = os.environ.get("ENABLE_LLM_EDITOR", "false").lower() == "true"
BenchMode = Literal["swe", "swt", "swt-ci"]

# Global variable to track dataset type
DATASET_TYPE = "SWE-bench"


class EvalException(Exception):
    """Exception raised during evaluation."""

    pass


class EvalMetadata:
    """Metadata for evaluation."""

    def __init__(
        self,
        llm_config,
        dataset_name,
        agent_class,
        max_iterations,
        eval_note,
        eval_output_dir,
        details=None,
    ):
        self.llm_config = llm_config
        self.dataset_name = dataset_name
        self.agent_class = agent_class
        self.max_iterations = max_iterations
        self.eval_note = eval_note
        self.eval_output_dir = eval_output_dir
        self.details = details or {}


class EvalOutput:
    """Output from evaluation."""

    def __init__(
        self, instance_id, git_patch, instruction, metadata, history, error=None
    ):
        self.instance_id = instance_id
        self.git_patch = git_patch
        self.instruction = instruction
        self.metadata = metadata
        self.history = history
        self.error = error

    def to_dict(self):
        """Convert to dictionary format matching example_.json structure."""
        return {
            "instance_id": self.instance_id,
            "test_result": {"git_patch": self.git_patch},
            "instruction": self.instruction,
            "metadata": self.metadata,
            "history": self.history,
        }


class LLMConfig:
    """LLM configuration."""

    def __init__(self, model, api_key, base_url=None, temperature=0):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.log_completions = True
        self.modify_params = False
        # Additional fields to match example_.json structure
        self.api_version = None
        self.aws_access_key_id = None
        self.aws_secret_access_key = None
        self.aws_region_name = None
        self.openrouter_site_url = None
        self.openrouter_app_name = None
        self.num_retries = None
        self.retry_multiplier = None
        self.retry_min_wait = None
        self.retry_max_wait = None
        self.timeout = None
        self.max_message_chars = None
        self.top_p = None
        self.top_k = None
        self.custom_llm_provider = None
        self.max_input_tokens = None
        self.max_output_tokens = None
        self.input_cost_per_token = None
        self.output_cost_per_token = None
        self.ollama_base_url = None
        self.drop_params = None
        self.disable_vision = None
        self.caching_prompt = None
        self.log_completions_folder = None
        self.custom_tokenizer = None
        self.native_tool_calling = None
        self.reasoning_effort = None
        self.seed = None
        self.safety_settings = None

    def to_dict(self):
        """Convert to dictionary format."""
        return {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_version": self.api_version,
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "aws_region_name": self.aws_region_name,
            "openrouter_site_url": self.openrouter_site_url,
            "openrouter_app_name": self.openrouter_app_name,
            "num_retries": self.num_retries,
            "retry_multiplier": self.retry_multiplier,
            "retry_min_wait": self.retry_min_wait,
            "retry_max_wait": self.retry_max_wait,
            "timeout": self.timeout,
            "max_message_chars": self.max_message_chars,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "custom_llm_provider": self.custom_llm_provider,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "input_cost_per_token": self.input_cost_per_token,
            "output_cost_per_token": self.output_cost_per_token,
            "ollama_base_url": self.ollama_base_url,
            "drop_params": self.drop_params,
            "modify_params": self.modify_params,
            "disable_vision": self.disable_vision,
            "caching_prompt": self.caching_prompt,
            "log_completions": self.log_completions,
            "log_completions_folder": self.log_completions_folder,
            "custom_tokenizer": self.custom_tokenizer,
            "native_tool_calling": self.native_tool_calling,
            "reasoning_effort": self.reasoning_effort,
            "seed": self.seed,
            "safety_settings": self.safety_settings,
        }


def set_dataset_type(dataset_name: str) -> str:
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


def get_instruction(
    instance: pd.Series, metadata: EvalMetadata, workspace_path: str
) -> str:
    """Generate instruction for the agent."""
    workspace_dir_name = _get_workspace_dir_name(instance)
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

    if RUN_WITH_BROWSING:
        instruction += (
            "<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web. </IMPORTANT!>\n"
        )

    return instruction


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


def process_instance_simplified(
    instance: pd.Series, metadata: EvalMetadata
) -> EvalOutput:
    """Process a single instance using the simplified SDK approach."""
    logger.info(f"Starting evaluation for instance {instance.instance_id}")

    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_workspace:
        try:
            # Setup workspace
            workspace_path = setup_workspace(instance, temp_workspace)
            initialize_workspace(workspace_path, instance)

            # Configure LLM
            api_key = os.getenv("LITELLM_API_KEY")
            if not api_key:
                raise EvalException("LITELLM_API_KEY environment variable is not set")

            llm_config = metadata.llm_config
            llm = LLM(
                model=llm_config.model,
                base_url=llm_config.base_url or "https://llm-proxy.eval.all-hands.dev",
                api_key=SecretStr(api_key),
                temperature=getattr(llm_config, "temperature", 0.0),
            )

            # Setup tools with the workspace
            tools: list[Tool] = [
                BashTool(working_dir=workspace_path),
                FileEditorTool(),
            ]

            # Create agent
            agent = Agent(llm=llm, tools=tools)

            # Collect full conversation history
            conversation_history = []

            def conversation_callback(event: Event):
                # Convert event to dictionary format matching example_.json structure
                event_dict = {
                    "id": getattr(event, "id", str(uuid.uuid4())),
                    "timestamp": getattr(
                        event, "timestamp", datetime.now().isoformat()
                    ),
                    "source": getattr(event, "source", "agent"),
                    "message": getattr(event, "message", ""),
                }

                # Add event-specific fields with proper serialization
                if hasattr(event, "action") and event.action:
                    # Convert action object to dictionary
                    action = event.action
                    if hasattr(action, "model_dump"):
                        event_dict["action"] = action.model_dump()
                    elif hasattr(action, "dict"):
                        event_dict["action"] = action.dict()
                    else:
                        # Fallback: convert to string representation
                        event_dict["action"] = {
                            "action": str(type(action).__name__),
                            "args": getattr(action, "args", {}),
                            "content": getattr(action, "content", ""),
                        }

                if hasattr(event, "args"):
                    event_dict["args"] = event.args
                if hasattr(event, "observation") and event.observation:
                    # Convert observation object to dictionary
                    obs = event.observation
                    if hasattr(obs, "model_dump"):
                        event_dict["observation"] = obs.model_dump()
                    elif hasattr(obs, "dict"):
                        event_dict["observation"] = obs.dict()
                    else:
                        # Fallback: convert to string representation
                        event_dict["observation"] = {
                            "observation": str(type(obs).__name__),
                            "content": getattr(obs, "content", ""),
                            "success": getattr(obs, "success", True),
                        }

                if hasattr(event, "content"):
                    event_dict["content"] = event.content
                if hasattr(event, "cause"):
                    event_dict["cause"] = event.cause
                if hasattr(event, "extras"):
                    event_dict["extras"] = event.extras

                # Add tool call metadata if available
                if hasattr(event, "tool_call_metadata"):
                    event_dict["tool_call_metadata"] = event.tool_call_metadata

                # Add LLM metrics if available
                if hasattr(event, "llm_metrics"):
                    event_dict["llm_metrics"] = event.llm_metrics

                conversation_history.append(event_dict)

            # Create conversation with callback
            conversation = Conversation(agent=agent, callbacks=[conversation_callback])

            # Get instruction
            instruction = get_instruction(instance, metadata, workspace_path)

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
                        ImageContent(image_urls=image_urls),
                    ],
                )
            else:
                message = Message(role="user", content=[TextContent(text=instruction)])

            # Send message and run conversation
            conversation.send_message(message)
            conversation.run()

            logger.info(
                f"Conversation completed with {len(conversation_history)} events"
            )

            # Get git patch
            git_patch = get_git_patch(workspace_path)

            # Create complete metadata structure
            complete_metadata = {
                "agent_class": metadata.agent_class,
                "llm_config": metadata.llm_config.to_dict(),
                "agent_config": metadata.agent_class,  # Same as agent_class for now
                "max_iterations": metadata.max_iterations,
                "eval_output_dir": metadata.eval_output_dir,
                "start_time": datetime.now().isoformat(),
                "git_commit": "unknown",  # Could be extracted from git if needed
                "dataset": metadata.dataset_name,
                "data_split": "test",  # Default, could be parameterized
                "details": metadata.details,
                "condenser_config": {
                    "type": "none"  # Default, could be parameterized
                },
            }

            logger.info(f"Completed evaluation for instance {instance.instance_id}")
            logger.info(f"Git patch length: {len(git_patch)} characters")

            return EvalOutput(
                instance.instance_id,
                git_patch,
                instruction,
                complete_metadata,
                conversation_history,
            )

        except Exception as e:
            logger.error(f"Error processing instance {instance.instance_id}: {str(e)}")
            # Create minimal metadata for error case
            error_metadata = {
                "agent_class": metadata.agent_class,
                "llm_config": metadata.llm_config.to_dict(),
                "agent_config": metadata.agent_class,
                "max_iterations": metadata.max_iterations,
                "eval_output_dir": metadata.eval_output_dir,
                "start_time": datetime.now().isoformat(),
                "git_commit": "unknown",
                "dataset": metadata.dataset_name,
                "data_split": "test",
                "details": metadata.details,
                "condenser_config": {"type": "none"},
            }
            return EvalOutput(
                instance.instance_id,
                "",
                "Error occurred",
                error_metadata,
                [],
                error=str(e),
            )


def get_evaluation_parser():
    """Get argument parser for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--agent-cls",
        dest="agent_cls",
        type=str,
        default=os.environ.get("AGENT_CLS", "CodeActAgent"),
    )
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
    llm_config,
    dataset_name,
    agent_class,
    max_iterations,
    eval_note,
    eval_output_dir,
    details=None,
):
    """Create evaluation metadata."""
    return EvalMetadata(
        llm_config,
        dataset_name,
        agent_class,
        max_iterations,
        eval_note,
        eval_output_dir,
        details,
    )


def construct_eval_output_dir(
    base_dir, dataset_name, agent_class, model, max_iterations, eval_note
):
    """Construct the structured evaluation output directory path."""
    # Format: eval_out/<dataset>-<split>/<agent_config>/<llm>_maxiter_<maxiter>_N_<version>-<hint>-<exp_name>-run_<run_number>

    # Extract model name (remove provider prefixes if any)
    model_name = model.replace("claude-3-5-sonnet-latest", "claude-sonnet-4-20250514")

    # Create agent config directory name
    agent_config = agent_class

    # Create LLM config string
    llm_config_str = f"{model_name}_maxiter_{max_iterations}"

    # Add version and note information
    version = "v1"  # Default version
    hint_status = "no-hint"  # Default hint status

    if eval_note:
        llm_config_str += f"_N_{version}-{hint_status}-{eval_note}-run_1"
    else:
        llm_config_str += f"_N_{version}-{hint_status}-run_1"

    # Construct full path
    eval_output_dir = os.path.join(base_dir, dataset_name, agent_config, llm_config_str)

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
        try:
            logger.info(f"Processing instance {instance.instance_id}")
            result = process_instance_simplified(instance, metadata)

            # Save result using the complete format
            result_dict = result.to_dict()
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

        except Exception as e:
            logger.error(f"Failed to process instance {instance.instance_id}: {str(e)}")
            # Create error result in complete format
            error_result = {
                "instance_id": instance.instance_id,
                "test_result": {"git_patch": ""},
                "instruction": "Error occurred",
                "metadata": {
                    "agent_class": metadata.agent_class,
                    "llm_config": metadata.llm_config.to_dict(),
                    "agent_config": metadata.agent_class,
                    "max_iterations": metadata.max_iterations,
                    "eval_output_dir": metadata.eval_output_dir,
                    "start_time": datetime.now().isoformat(),
                    "git_commit": "unknown",
                    "dataset": metadata.dataset_name,
                    "data_split": "test",
                    "details": metadata.details,
                    "condenser_config": {"type": "none"},
                },
                "history": [],
                "error": str(e),
            }
            results.append(error_result)

            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_file, "a") as f:
                json_line = json.dumps(error_result) + "\n"
                f.write(json_line)
                f.flush()
                logger.info("Successfully wrote error result to output file")


if __name__ == "__main__":
    parser = get_evaluation_parser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench",
        help="data set to evaluate on, either full-test or lite-test",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="split to evaluate on",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="swe",
        choices=["swe", "swt", "swt-ci"],
        help="mode to run the evaluation, either 'swe', 'swt', or 'swt-ci'",
    )

    args, _ = parser.parse_known_args()

    # Load dataset
    dataset = load_dataset(args.dataset, split=args.split)
    set_dataset_type(args.dataset)

    swe_bench_tests = filter_dataset(dataset.to_pandas(), "instance_id")
    logger.info(
        f"Loaded dataset {args.dataset} with split {args.split}: {len(swe_bench_tests)} tasks"
    )

    # Create LLM config
    llm_config = LLMConfig(
        model=args.model,
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url="https://llm-proxy.eval.all-hands.dev",
        temperature=0,
    )

    if not llm_config.api_key:
        raise ValueError("LITELLM_API_KEY environment variable is not set")

    details = {"mode": args.mode}
    dataset_description = (
        args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__")
    )

    # Construct proper structured output directory path
    structured_output_dir = construct_eval_output_dir(
        args.eval_output_dir,
        dataset_description,
        args.agent_cls,
        args.model,
        args.max_iterations,
        args.eval_note,
    )

    metadata = make_metadata(
        llm_config,
        dataset_description,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        structured_output_dir,
        details=details,
    )

    # Create output directory and file
    os.makedirs(metadata.eval_output_dir, exist_ok=True)
    output_file = os.path.join(metadata.eval_output_dir, "output.jsonl")
    print(f"### OUTPUT FILE: {output_file} ###")

    # Prepare dataset
    instances = prepare_dataset(swe_bench_tests, output_file, args.eval_n_limit)

    # Run evaluation
    run_evaluation_simplified(instances, metadata, output_file)

    logger.info("Evaluation completed!")
