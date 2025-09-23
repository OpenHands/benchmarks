import asyncio
import json
import os
import re
import tempfile
from typing import Any

import pandas as pd
from datasets import load_dataset

# import openhands.agenthub  # Not available in SDK
from benchmarks.multi_swe_bench.resource.mapping import (
    get_instance_resource_factor,
)
from benchmarks.utils.dataset import filter_dataset, prepare_dataset
from benchmarks.utils.run_evaluation import make_metadata
from benchmarks.utils.runtime import Runtime as BenchmarkRuntime
from benchmarks.utils.shared import EvalException, EvalMetadata, EvalOutput

# from openhands.controller.state.state import State  # Not available in SDK
# from openhands.core.config import (  # Not available in SDK
#     AgentConfig,
#     OpenHandsConfig,
#     SandboxConfig,
#     get_evaluation_parser,
#     get_llm_config_arg,
# )
from openhands.sdk import get_logger

# from openhands.core.main import create_runtime, run_controller  # Not available in SDK
# from openhands.events.action import CmdRunAction, FileReadAction, MessageAction  # N/A
# from openhands.events.observation import CmdOutputObservation, ErrorObservation  # N/A
# from openhands.events.serialization.event import event_to_dict  # Not available in SDK
# from openhands.runtime.base import Runtime  # Not available in SDK
# from openhands.utils.async_utils import call_async_from_sync  # Not available in SDK
# from openhands.utils.shutdown_listener import sleep_if_should_continue  # N/A
# SDK equivalents
from openhands.tools import (
    ExecuteBashObservation,
)


logger = get_logger(__name__)


# Placeholder classes and functions to replace missing OpenHands functionality
class State:
    """Placeholder for OpenHands State class"""

    def __init__(self):
        self.history = []
        self.last_user_message = ""


class AgentConfig:
    """Placeholder for OpenHands AgentConfig class"""

    def __init__(
        self,
        agent_cls=None,
        llm_config=None,
        max_iterations=None,
        enable_jupyter=False,
        enable_browsing=False,
        enable_llm_editor=False,
        condenser=None,
        enable_prompt_extensions=False,
    ):
        self.agent_cls = agent_cls
        self.llm_config = llm_config
        self.max_iterations = max_iterations
        self.enable_jupyter = enable_jupyter
        self.enable_browsing = enable_browsing
        self.enable_llm_editor = enable_llm_editor
        self.condenser = condenser
        self.enable_prompt_extensions = enable_prompt_extensions


class OpenHandsConfig:
    """Placeholder for OpenHands OpenHandsConfig class"""

    def __init__(self):
        self.agent = None
        self.sandbox = None
        self.llm = None

    def set_llm_config(self, llm_config):
        self.llm = llm_config


class SandboxConfig:
    """Placeholder for OpenHands SandboxConfig class"""

    def __init__(self):
        self.base_container_image = None
        self.enable_auto_lint = False
        self.use_host_network = False
        self.platform = None
        self.remote_runtime_resource_factor = 1.0


class Runtime:
    """Placeholder for OpenHands Runtime class"""

    def __init__(self, config):
        self.config = config

    async def connect(self):
        pass

    def run_action(self, action):
        # Simplified implementation - just return a success observation
        return ExecuteBashObservation(
            content="Command executed successfully", exit_code=0, command_id=-1
        )


class CmdRunAction:
    """Placeholder for OpenHands CmdRunAction class"""

    def __init__(self, command):
        self.command = command


class FileReadAction:
    """Placeholder for OpenHands FileReadAction class"""

    def __init__(self, path):
        self.path = path


class MessageAction:
    """Placeholder for OpenHands MessageAction class"""

    def __init__(self, content):
        self.content = content


class CmdOutputObservation:
    """Placeholder for OpenHands CmdOutputObservation class"""

    def __init__(self, content="", exit_code=0):
        self.content = content
        self.exit_code = exit_code


class ErrorObservation:
    """Placeholder for OpenHands ErrorObservation class"""

    def __init__(self, content=""):
        self.content = content


def get_evaluation_parser():
    """Placeholder for OpenHands get_evaluation_parser function"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-cls", type=str, default="CodeActAgent")
    parser.add_argument("--llm-config", type=str, default="")
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--eval-note", type=str, default="")
    parser.add_argument("--eval-output-dir", type=str, default="./outputs")
    parser.add_argument("--data-split", type=str, default="test")
    parser.add_argument("--max-instances", type=int, default=None)
    return parser


def get_llm_config_arg(llm_config_str):
    """Placeholder for OpenHands get_llm_config_arg function"""
    # Return a simple dict for now
    return {"model": "gpt-4", "api_key": ""}


def create_runtime(config):
    """Placeholder for OpenHands create_runtime function"""
    return Runtime(config)


async def run_controller(config, runtime, initial_user_action):
    """Placeholder for OpenHands run_controller function"""
    # Simplified implementation - just return a basic state
    state = State()
    state.history = [initial_user_action]
    state.last_user_message = initial_user_action.content
    return state


def event_to_dict(event):
    """Placeholder for OpenHands event_to_dict function"""
    return {"type": type(event).__name__, "content": getattr(event, "content", "")}


def call_async_from_sync(coro):
    """Placeholder for OpenHands call_async_from_sync function"""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def sleep_if_should_continue(seconds):
    """Placeholder for OpenHands sleep_if_should_continue function"""
    import time

    time.sleep(seconds)


# Placeholder for openhands.agenthub module
class OpenHandsAgentHub:
    class Agent:
        @staticmethod
        def get_cls(agent_cls_name):
            # Return a placeholder agent class
            return type(agent_cls_name, (), {})


class OpenHandsModule:
    def __init__(self):
        self.agenthub = OpenHandsAgentHub()


openhands = OpenHandsModule()


USE_HINT_TEXT = os.environ.get("USE_HINT_TEXT", "false").lower() == "true"
USE_INSTANCE_IMAGE = os.environ.get("USE_INSTANCE_IMAGE", "true").lower() == "true"
RUN_WITH_BROWSING = os.environ.get("RUN_WITH_BROWSING", "false").lower() == "true"

# TODO: migrate all swe-bench docker to ghcr.io/openhands
# TODO: 适应所有的语言
DOCKER_IMAGE_PREFIX = os.environ.get("EVAL_DOCKER_IMAGE_PREFIX", "")
LANGUAGE = os.environ.get("LANGUAGE", "python")
logger.info(f"Using docker image prefix: {DOCKER_IMAGE_PREFIX}")


def codeact_user_response(state: State) -> str:
    """
    Provide automated user response for CodeActAgent during evaluation.

    This function is called when the agent requests user input during automated
    evaluation. For SWE-bench evaluation, we typically want the agent to proceed
    without user intervention.
    """
    # Return a generic response that encourages the agent to continue
    return "Please continue with your task. You have all the information you need."


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    "CodeActAgent": codeact_user_response,
}


def _get_swebench_workspace_dir_name(instance: pd.Series) -> str:
    return f"{instance.repo}__{instance.version}".replace("/", "__")


def get_default_sandbox_config_for_eval():
    """Get default sandbox configuration for evaluation."""
    return SandboxConfig()


def get_openhands_config_for_eval(metadata, enable_browser, runtime, sandbox_config):
    """Get OpenHands configuration for evaluation."""
    config = OpenHandsConfig()
    config.sandbox = sandbox_config
    return config


def update_llm_config_for_completions_logging(llm_config, output_dir, instance_id):
    """Update LLM config for completions logging."""
    return llm_config


def get_instruction(instance: pd.Series, metadata: EvalMetadata):
    """Get instruction using utils get_instruction with language prompts."""
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    workspace_path = f"/workspace/{workspace_dir_name}"

    # Construct the prompt path based on the language
    prompt_path = os.path.join(
        os.path.dirname(__file__), "prompts", f"{LANGUAGE.lower()}.j2"
    )

    # Use the utils get_instruction function
    from benchmarks.utils.run_evaluation import get_instruction as utils_get_instruction

    instruction = utils_get_instruction(instance, metadata, workspace_path, prompt_path)

    if instruction and RUN_WITH_BROWSING:
        instruction += (
            "<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web. </IMPORTANT!>\n"
        )
    return instruction


# TODO: 适应所有的语言
# def get_instance_docker_image(instance_id: str) -> str:
#     image_name = 'sweb.eval.x86_64.' + instance_id
#     if LANGUAGE == 'python':
#         image_name = image_name.replace(
#             '__', '_s_'
#         )  # to comply with docker image naming convention
#         return (DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name).lower()
#     else:
#         return image_name.lower()  # 加载本地的
def get_instance_docker_image(instance: pd.Series):
    if LANGUAGE == "python":
        image_name = "sweb.eval.x86_64." + str(instance["instance_id"])
        image_name = image_name.replace(
            "__", "_s_"
        )  # to comply with docker image naming convention
        return (DOCKER_IMAGE_PREFIX.rstrip("/") + "/" + image_name).lower()
    else:
        container_name = str(instance.get("repo", "")).lower()
        container_name = container_name.replace("/", "_m_")
        instance_id = str(instance.get("instance_id", ""))
        tag_suffix = instance_id.split("-")[-1] if instance_id else ""
        container_tag = f"pr-{tag_suffix}"
        # pdb.set_trace()
        return f"mswebench/{container_name}:{container_tag}"
        # return "kong/insomnia:pr-8284"
        # return "'sweb.eval.x86_64.local_insomnia"
        # return "local_insomnia_why"
        # return "local/kong-insomnia:pr-8117"


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    SWE_BENCH_CONTAINER_IMAGE = "ghcr.io/opendevin/eval-swe-bench:full-v1.2.1"
    if USE_INSTANCE_IMAGE:
        # We use a different instance image for the each instance of swe-bench eval
        # base_container_image = get_instance_docker_image(instance['instance_id'])
        base_container_image = get_instance_docker_image(instance)
        logger.info(
            f"Using instance container image: {base_container_image}. "
            f"Please make sure this image exists. "
            f"Submit an issue on https://github.com/All-Hands-AI/OpenHands "
            f"if you run into any issues."
        )
    else:
        base_container_image = SWE_BENCH_CONTAINER_IMAGE
        logger.info(f"Using swe-bench container image: {base_container_image}")

    sandbox_config = get_default_sandbox_config_for_eval()
    # sandbox_config.base_container_image = base_container_image
    # sandbox_config.enable_auto_lint = True
    # sandbox_config.use_host_network = False
    # Add platform to the sandbox config to solve issue 4401
    # sandbox_config.platform = "linux/amd64"
    if metadata.dataset is None:
        raise ValueError("Dataset is required")
    # sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
    #     dataset_name=metadata.dataset,
    #     instance_id=str(instance["instance_id"]),
    # )

    config = get_openhands_config_for_eval(
        metadata=metadata,
        enable_browser=RUN_WITH_BROWSING,
        runtime=os.environ.get("RUNTIME", "docker"),
        sandbox_config=sandbox_config,
    )
    # config.set_llm_config(
    #     update_llm_config_for_completions_logging(
    #         metadata.llm_config, metadata.eval_output_dir, instance["instance_id"]
    #     )
    # )
    # agent_config = AgentConfig(
    #     enable_jupyter=False,
    #     enable_browsing=RUN_WITH_BROWSING,
    #     enable_llm_editor=False,
    #     condenser=metadata.condenser_config,
    #     enable_prompt_extensions=False,
    # )
    # config.set_agent_config(agent_config)
    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info("-" * 30)
    logger.info("BEGIN Runtime Initialization Fn")
    logger.info("-" * 30)
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    obs: CmdOutputObservation

    REPO_NAME = str(instance["repo"]).split("/")[-1]
    # Set instance id
    cmd_parts = [
        f"echo 'export SWE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc",
        "echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc",
        "echo \"alias git='git --no-pager'\" >> ~/.bashrc",
        f"echo 'export REPO_NAME={REPO_NAME}' >> ~/.bashrc",
    ]
    action = CmdRunAction(command=" && ".join(cmd_parts))
    # action.set_hard_timeout(600)
    logger.info(action, extra={"msg_type": "ACTION"})
    obs = runtime.run_action(action)
    logger.info(obs, extra={"msg_type": "OBSERVATION"})
    assert obs.exit_code == 0, f"Failed to export SWE_INSTANCE_ID: {str(obs)}"
    # pdb.set_trace()
    action = CmdRunAction(command="""export USER=$(whoami); echo USER=${USER} """)
    # action.set_hard_timeout(600)
    logger.info(action, extra={"msg_type": "ACTION"})
    obs = runtime.run_action(action)
    logger.info(obs, extra={"msg_type": "OBSERVATION"})
    assert obs.exit_code == 0, f"Failed to export USER: {str(obs)}"

    if USE_INSTANCE_IMAGE:
        # inject the init script
        script_dir = os.path.dirname(__file__)

        # inject the instance info
        action = CmdRunAction(command="mkdir -p /swe_util/eval_data/instances")
        # action.set_hard_timeout(600)
        logger.info(action, extra={"msg_type": "ACTION"})
        obs = runtime.run_action(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert obs.exit_code == 0, (
            f"Failed to create /swe_util/eval_data/instances: {str(obs)}"
        )

        swe_instance_json_name = "swe-bench-instance.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            # Construct the full path for the desired file name within temp dir
            temp_file_path = os.path.join(temp_dir, swe_instance_json_name)
            # Write to the file with the desired name within the temporary directory
            with open(temp_file_path, "w") as f:
                if not isinstance(instance, dict):
                    json.dump([instance.to_dict()], f)
                else:
                    json.dump([instance], f)

            # Copy the file to the desired location
            # runtime.copy_to(temp_file_path, "/swe_util/eval_data/instances/")

        # inject the instance swe entry
        # runtime.copy_to(
        #     str(os.path.join(script_dir, "scripts/setup/instance_swe_entry.sh")),
        #     "/swe_util/",
        # )
        action = CmdRunAction(command="cat ~/.bashrc")
        # action.set_hard_timeout(600)
        logger.info(action, extra={"msg_type": "ACTION"})
        obs = runtime.run_action(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert obs.exit_code == 0, f"Failed to cat ~/.bashrc: {str(obs)}"

        action = CmdRunAction(command="source ~/.bashrc")
        # action.set_hard_timeout(600)
        logger.info(action, extra={"msg_type": "ACTION"})
        obs = runtime.run_action(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        if isinstance(obs, ErrorObservation):
            logger.error(f"Failed to source ~/.bashrc: {str(obs)}")
        # assert obs.exit_code == 0, f"Failed to source ~/.bashrc: {str(obs)}"

        action = CmdRunAction(command="source /swe_util/instance_swe_entry.sh")
        # action.set_hard_timeout(600)
        logger.info(action, extra={"msg_type": "ACTION"})
        obs = runtime.run_action(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert obs.exit_code == 0, (
            f"Failed to source /swe_util/instance_swe_entry.sh: {str(obs)}"
        )
    else:
        action = CmdRunAction(command="source /swe_util/swe_entry.sh")
        # action.set_hard_timeout(1800)
        logger.info(action, extra={"msg_type": "ACTION"})
        obs = runtime.run_action(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert obs.exit_code == 0, (
            f"Failed to source /swe_util/swe_entry.sh: {str(obs)}"
        )

    action = CmdRunAction(command=f"cd /workspace/{workspace_dir_name}")
    # action.set_hard_timeout(600)
    logger.info(action, extra={"msg_type": "ACTION"})
    obs = runtime.run_action(action)
    logger.info(obs, extra={"msg_type": "OBSERVATION"})
    assert obs.exit_code == 0, (
        f"Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}"
    )

    action = CmdRunAction(command="git reset --hard")
    # action.set_hard_timeout(600)
    logger.info(action, extra={"msg_type": "ACTION"})
    obs = runtime.run_action(action)
    logger.info(obs, extra={"msg_type": "OBSERVATION"})
    assert obs.exit_code == 0, f"Failed to git reset --hard: {str(obs)}"

    cmd = (
        'for remote_name in $(git remote); do git remote remove "${remote_name}"; done'
    )
    action = CmdRunAction(command=cmd)
    # action.set_hard_timeout(600)
    logger.info(action, extra={"msg_type": "ACTION"})
    obs = runtime.run_action(action)
    logger.info(obs, extra={"msg_type": "OBSERVATION"})
    assert obs.exit_code == 0, f"Failed to remove git remotes: {str(obs)}"
    # TODO:这里看看需不需要判断其他语言的环境
    # action = CmdRunAction(command='which python')
    # action.set_hard_timeout(600)
    # logger.info(action, extra={'msg_type': 'ACTION'})
    # obs = runtime.run_action(action)
    # logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    # assert
    #     obs.exit_code == 0 and 'testbed' in obs.content,
    #     f'Expected to find python interpreter from testbed, but got: {str(obs)}',
    # )

    logger.info("-" * 30)
    logger.info("END Runtime Initialization Fn")
    logger.info("-" * 30)


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,  # used to get the workspace_dir_name
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info("-" * 30)
    logger.info("BEGIN Runtime Completion Fn")
    logger.info("-" * 30)
    obs: CmdOutputObservation
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)

    action = CmdRunAction(command=f"cd /workspace/{workspace_dir_name}")
    # action.set_hard_timeout(600)
    logger.info(action, extra={"msg_type": "ACTION"})
    obs = runtime.run_action(action)
    logger.info(obs, extra={"msg_type": "OBSERVATION"})

    if obs.exit_code == -1:
        # The previous command is still running
        # We need to kill previous command
        logger.info("The previous command is still running, trying to kill it...")
        action = CmdRunAction(command="C-c")
        obs = runtime.run_action(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})

        # Then run the command again
        action = CmdRunAction(command=f"cd /workspace/{workspace_dir_name}")
        # action.set_hard_timeout(600)
        logger.info(action, extra={"msg_type": "ACTION"})
        obs = runtime.run_action(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})

    assert isinstance(obs, CmdOutputObservation) and obs.exit_code == 0, (
        f"Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}"
    )

    action = CmdRunAction(command='git config --global core.pager ""')
    # action.set_hard_timeout(600)
    logger.info(action, extra={"msg_type": "ACTION"})
    obs = runtime.run_action(action)
    logger.info(obs, extra={"msg_type": "OBSERVATION"})
    assert isinstance(obs, CmdOutputObservation) and obs.exit_code == 0, (
        f'Failed to git config --global core.pager "": {str(obs)}'
    )

    action = CmdRunAction(command="git add -A")
    # action.set_hard_timeout(600)
    logger.info(action, extra={"msg_type": "ACTION"})
    obs = runtime.run_action(action)
    logger.info(obs, extra={"msg_type": "OBSERVATION"})
    assert isinstance(obs, CmdOutputObservation) and obs.exit_code == 0, (
        f"Failed to git add -A: {str(obs)}"
    )

    # 删除二进制文件
    action = CmdRunAction(
        command="""
        for file in $(git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" \\
                      | cut -c4-); do
            if [ -f "$file" ] && (file "$file" | grep -q "executable" \\
                                   || git check-attr binary "$file" \\
                                      | grep -q "binary: set"); then
                git rm -f "$file" 2>/dev/null || rm -f "$file"
                echo "Removed: $file"
            fi
        done
        """
    )
    # action.set_hard_timeout(600)
    logger.info(action, extra={"msg_type": "ACTION"})
    obs = runtime.run_action(action)
    logger.info(obs, extra={"msg_type": "OBSERVATION"})
    assert isinstance(obs, CmdOutputObservation) and obs.exit_code == 0, (
        f"Failed to remove binary files: {str(obs)}"
    )

    # pdb.set_trace()

    n_retries = 0
    git_patch = None
    while n_retries < 5:
        cmd = f"git diff --no-color --cached {instance['base_commit']} > patch.diff"
        action = CmdRunAction(command=cmd)
        # action.set_hard_timeout(max(300 + 100 * n_retries, 600))
        logger.info(action, extra={"msg_type": "ACTION"})
        obs = runtime.run_action(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        n_retries += 1
        if isinstance(obs, CmdOutputObservation):
            if obs.exit_code == 0:
                # git_patch = obs.content.strip()
                break
            else:
                logger.info("Failed to get git diff, retrying...")
                sleep_if_should_continue(10)
        elif isinstance(obs, ErrorObservation):
            logger.error(f"Error occurred: {obs.content}. Retrying...")
            sleep_if_should_continue(10)
        else:
            assert False, f"Unexpected observation type: {str(obs)}"

    action = FileReadAction(path="patch.diff")
    # action.set_hard_timeout(max(300 + 100 * n_retries, 600))
    logger.info(action, extra={"msg_type": "ACTION"})
    obs = runtime.run_action(action)
    git_patch = obs.content
    # pdb.set_trace()

    assert git_patch is not None, "Failed to get git diff (None)"

    logger.info("-" * 30)
    logger.info("END Runtime Completion Fn")
    logger.info("-" * 30)
    return {"git_patch": git_patch}


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    config = get_config(instance, metadata)

    # Setup the logger properly, so you can run multi-processing to parallelize
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, "infer_logs")
        logger.info(
            f"Starting evaluation for instance {instance.instance_id} "
            f"with log dir {log_dir}."
        )
    else:
        logger.info(f"Starting evaluation for instance {instance.instance_id}.")

    # Increase resource_factor with increasing attempt_id
    if runtime_failure_count > 0:
        # config.sandbox.remote_runtime_resource_factor = min(
        #     config.sandbox.remote_runtime_resource_factor * (2**runtime_failure_count),
        #     8,
        # )
        logger.warning(
            f"This is the {runtime_failure_count + 1}th attempt for instance "
            f"{instance.instance_id}, setting resource factor to "
            # f"{config.sandbox.remote_runtime_resource_factor}"
        )
    # pdb.set_trace()
    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

    try:
        # # initialize_runtime(runtime, instance)

        instruction = get_instruction(instance, metadata)

        # Here's how you can run the agent (similar to the `main` function)
        # and get the final task state
        state: State | None = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=MessageAction(content=instruction),
                runtime=runtime,
                # fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[
                #     metadata.agent_class
                # ],
            )
        )

        # if fatal error, throw EvalError to trigger re-run
        # if state.last_error and "fatal" in state.last_error.lower():
        #     raise EvalException("Fatal error detected: " + state.last_error)

        # ======= THIS IS SWE-Bench specific =======
        # Get git patch
        # # return_val = complete_runtime(runtime, instance)
        git_patch = ""  # return_val["git_patch"]
        logger.info(
            f"Got git diff for instance {instance.instance_id}:\n"
            f"--------\n{git_patch}\n--------"
        )
    finally:
        # runtime.close()
        pass
    # ==========================================

    # ======= Attempt to evaluate the agent's edits =======
    # we use eval_infer.sh to evaluate the agent's edits, not here
    # because the agent may alter the environment / testcases
    # remove binary diffs
    def remove_binary_diffs(patch_text):
        lines = patch_text.splitlines()
        cleaned_lines = []
        block = []
        is_binary_block = False

        for line in lines:
            if line.startswith("diff --git "):
                if block and not is_binary_block:
                    cleaned_lines.extend(block)
                block = [line]
                is_binary_block = False
            elif "Binary files" in line:
                is_binary_block = True
                block.append(line)
            else:
                block.append(line)

        if block and not is_binary_block:
            cleaned_lines.extend(block)
        return "\n".join(cleaned_lines)

    git_patch = remove_binary_diffs(git_patch)
    test_result = {
        "git_patch": git_patch,
    }

    # If you are working on some simpler benchmark that only evaluates the final
    # model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned
    # `state.history` and parse it for evaluation.
    if state is None:
        raise ValueError("State should not be None.")

    # NOTE: this is NO LONGER the event stream, but an agent history that
    # includes delegate agent's events
    histories = [event_to_dict(event) for event in state.history]
    metrics = {"total_cost": 0.0}

    # Save the output
    output = EvalOutput(
        instance_id=instance.instance_id,
        instruction=instruction,
        instance=instance.to_dict(),  # SWE Bench specific
        test_result=test_result,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=None,  # state.last_error if state and state.last_error else None,
    )
    return output


if __name__ == "__main__":
    # pdb.set_trace()
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
    args, _ = parser.parse_known_args()

    # NOTE: It is preferable to load datasets from huggingface datasets
    # and perform post-processing
    # so we don't need to manage file uploading to OpenHands's repo
    # dataset = load_dataset(args.dataset, split=args.split)
    # dataset = load_dataset(args.dataset)
    dataset = load_dataset("json", data_files=args.dataset)
    # # dataset = dataset[args.split]
    # swe_bench_tests = filter_dataset(
    #     dataset.to_pandas(), "instance_id", os.path.dirname(os.path.abspath(__file__))
    # )
    swe_bench_tests = []
    logger.info(
        f"Loaded dataset {args.dataset} with split {args.split}: "
        f"{len(swe_bench_tests)} tasks"
    )

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        # # llm_config.log_completions = True
        # modify_params must be False for evaluation purpose, for
        # reproducibility and accurancy of results
        # # llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f"Could not find LLM config: --llm_config {args.llm_config}")

    details = {}
    _agent_cls = openhands.agenthub.Agent.get_cls(args.agent_cls)

    match = re.search(r"Multi-SWE-bench/[^/]+/[^/]+", args.dataset)
    if match:
        dataset_description = match.group(0) + "-" + args.split.replace("/", "__")
    else:
        dataset_description = (
            args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__")
        )

    metadata = make_metadata(
        llm_config,
        dataset_description,
        args.max_iterations,
        args.eval_output_dir,
        details=details,
    )

    # Global variables for runtime methods
    global instances, output_file, results
    instances = None
    output_file = None
    results = []

    def initialize_runtime():
        """Initialize the runtime and retrieve instances to process."""
        global instances, output_file
        output_file = os.path.join(metadata.eval_output_dir, "output.jsonl")
        print(f"### OUTPUT FILE: {output_file} ###")

        # Prepare output file
        output_dir = os.path.dirname(output_file)
        if output_dir:  # Only create directory if dirname is not empty
            os.makedirs(output_dir, exist_ok=True)

        # Create empty output file
        with open(output_file, "w"):
            pass

        # Retrieve instances to process
        # instances = prepare_dataset(swe_bench_tests, output_file, args.eval_n_limit)
        instances = []

        # if len(instances) > 0 and not isinstance(
        #     instances["FAIL_TO_PASS"][instances["FAIL_TO_PASS"].index[0]], str
        # ):
        #     for col in ["PASS_TO_PASS", "FAIL_TO_PASS"]:
        #         instances[col] = instances[col].apply(lambda x: str(x))
        # if LANGUAGE == "java": ##TODO:适配多语言的版本
        #     for col in ['issue_numbers', 'created_at']:
        #         instances[col] = instances[col].apply(lambda x: str(x))

        return instances

    def process_instance_wrapper(instance):
        """Process a single instance using the existing process_instance function."""
        global results, output_file
        logger.info(f"Processing instance {instance.instance_id}")

        # Use the existing process_instance function
        result = process_instance(instance, metadata, True)

        # The existing process_instance function already handles writing to output file
        # so we don't need to do additional processing here
        return result

    def complete_runtime():
        """Complete the runtime - any cleanup if needed."""
        logger.info("Runtime completed successfully!")
        # Check if any instances reached maximum retries
        logger.info("Evaluation completed successfully")

    # Create and run the Runtime
    # runtime = BenchmarkRuntime(
    #     metadata=metadata,
    #     initialize_runtime=initialize_runtime,
    #     process_instance=process_instance_wrapper,
    #     complete_runtime=complete_runtime,
    # )
    runtime = None

    # # runtime.run()
