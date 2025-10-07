from __future__ import annotations

import os
import threading
import time
from typing import Any

import pandas as pd
from pydantic import SecretStr

from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.conversation_tools import get_git_patch_from_history, get_history
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_instruction,
    make_metadata,
    read_completed_instances,
    write_output_to_file,
)
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.shared import EvalMetadata
from openhands.sdk import LLM, Agent, RemoteWorkspace, Workspace, get_logger
from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation
from openhands.tools.preset.default import get_default_tools


logger = get_logger(__name__)


def run_evaluation(llm: Any, metadata: EvalMetadata, num_workers: int = 1):
    """
    Run evaluation using remote evaluation mode (agent server).

    Args:
        llm: LLM instance to use for evaluation
        metadata: EvalMetadata object containing evaluation configuration
        num_workers: Number of worker threads to use for parallel processing
    """
    logger.info("Running evaluation in REMOTE mode")
    logger.info(f"Using {num_workers} workers for parallel processing")
    workspace_path = "/workspace"

    # Shared state using closure variables instead of globals
    evaluation_state = {
        "instances": None,
        "output_file": None,
        "results": [],
        "agent": None,
        "llm_instance": llm,
    }

    def initialize_dataset_run():
        """Initialize the dataset run and return instances to process."""
        # Create agent
        evaluation_state["agent"] = Agent(
            llm=evaluation_state["llm_instance"],
            tools=get_default_tools(enable_browser=False),  # Disable browser tools
        )
        # Prepare output file
        evaluation_state["output_file"] = os.path.join(
            metadata.eval_output_dir or ".", "output.jsonl"
        )
        output_dir = os.path.dirname(evaluation_state["output_file"])
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Read existing completed instances instead of overwriting
        completed_instances = read_completed_instances(evaluation_state["output_file"])
        if completed_instances:
            logger.info(f"Found {len(completed_instances)} already completed instances")
        else:
            logger.info("No existing results found, starting fresh")
            # Create empty output file only if it doesn't exist
            if not os.path.exists(evaluation_state["output_file"]):
                with open(evaluation_state["output_file"], "w"):
                    pass

        # Retrieve instances to process, excluding completed ones
        evaluation_state["instances"] = get_dataset(
            metadata.dataset or "",
            metadata.data_split or "",
            evaluation_state["output_file"],
            metadata.eval_n_limit or 0,
            completed_instances,
        )
        # Add repo_path column by extracting repo name from org/repo format
        evaluation_state["instances"]["repo_path"] = (
            evaluation_state["instances"]["repo"]
            .str.split("/")
            .str[1]
            .apply(lambda name: os.path.join(workspace_path, name))
        )
        print(f"### OUTPUT FILE: {evaluation_state['output_file']} ###")
        return evaluation_state["instances"]

    def process_instance(instance):
        """Process a single instance using remote conversation."""
        logger.info(f"Processing instance: {instance.instance_id}")

        # Get instruction
        instruction = get_instruction(
            instance, metadata, workspace_path, metadata.prompt_path or ""
        )

        # Get the worker's server port (this will be set by the worker)
        worker_port = getattr(threading.current_thread(), "server_port", 8001)
        server_url = f"http://localhost:{worker_port}"

        conversation = None

        # Set up callback collection, like example 22
        received_events: list = []
        last_event_time = {"ts": time.time()}

        def event_callback(event) -> None:
            event_type = type(event).__name__
            logger.info(f"🔔 Callback received event: {event_type}\n{event}")
            received_events.append(event)
            last_event_time["ts"] = time.time()

        try:
            if evaluation_state["agent"] is None:
                raise ValueError("Agent cannot be None")
            workspace: RemoteWorkspace = Workspace(host=server_url)

            repo_url = f"https://github.com/{instance.repo}"
            # Clone repository
            repo_clone_output = workspace.execute_command(f"""
                mkdir -p {workspace_path} ;
                git clone {repo_url} {instance.repo_path} > /dev/null 2>&1 ;
                cd {instance.repo_path} ;
                git branch | grep -v "main" | xargs git branch -D > /dev/null 2>&1 ;
                git reset --hard {instance.base_commit} ;
            """)

            if repo_clone_output["exit_code"] != 0:
                stderr = repo_clone_output["stderr"]
                stdout = repo_clone_output["stdout"]
                logger.info(
                    f"Cloning failed with {repo_clone_output['exit_code']}: {stderr}"
                )
                logger.info(f"Cloning failed with stdout: {stdout}")
                raise ValueError("Cloning failed")
            else:
                stdout = repo_clone_output["stdout"]
                logger.info(f"Cloning repository succeeded with stdout: {stdout}")

                conversation = RemoteConversation(
                    agent=evaluation_state["agent"],
                    workspace=workspace,
                    callbacks=[event_callback],
                    max_iteration_per_run=metadata.max_iterations,
                )

                assert isinstance(conversation, RemoteConversation)

            # Send message and run with event streaming
            logger.info(f"Sending instruction to conversation: {instruction[:100]}...")
            conversation.send_message(instruction)

            # Add callback to log events as they happen
            def log_event(event):
                event_type = type(event).__name__
                event_content = getattr(
                    event,
                    "message",
                    getattr(event, "content", getattr(event, "action", str(event))),
                )
                logger.info(f"Event: {event_type} - {str(event_content)[:100]}")

            logger.info("Starting conversation.run()...")
            conversation.run()
            logger.info("Conversation.run() completed")
            history = get_history(conversation)

            # Retry git patch extraction up to 5 times before falling back to history
            git_patch = ""
            max_retries = 5
            for attempt in range(max_retries):
                logger.info(f"Git patch extraction attempt {attempt + 1}/{max_retries}")
                result = workspace.execute_command(
                    (
                        f"cd {instance.repo_path} ; "
                        "git config --global core.pager '' > /dev/null 2>&1 ; "
                        "git add -A > /dev/null 2>&1 ; "
                        f"git --no-pager diff --cached {instance.base_commit}"
                    )
                )
                logger.info(
                    f"Patch command execution (attempt {attempt + 1}): {result}"
                )
                exit_code = result["exit_code"]

                if result["exit_code"] != 0:
                    stderr = result["stderr"]
                    stdout = result["stdout"]
                    logger.info(f"Command failed with exit code {exit_code}: {stderr}")
                    logger.info(f"Command failed with stdout: {stdout}")
                else:
                    git_patch = result["stdout"]
                    if git_patch:
                        logger.info(
                            f"Successfully extracted git patch on attempt {attempt + 1}"
                        )
                        break
                    else:
                        logger.info(f"Git patch empty on attempt {attempt + 1}")

                # Wait a bit before retrying (except on last attempt)
                if attempt < max_retries - 1:
                    time.sleep(1)

            if not git_patch:
                logger.info("git_patch empty after all retries, searching history.")
                git_patch = get_git_patch_from_history(history)
                logger.info(f"Patch from history: {git_patch}")

            logger.info(f"Extracted git patch with {len(git_patch)} characters")

            # Extract results from conversation state
            logger.info("Starting result extraction from conversation state")
            from benchmarks.utils.shared import EvalOutput

            logger.info(f"Creating EvalOutput with: instance_id={instance.instance_id}")
            logger.info(f"Creating EvalOutput with: history_events={len(history)}")
            logger.info(f"Creating EvalOutput with: git_patch_length={len(git_patch)}")

            result = EvalOutput(
                instance_id=instance.instance_id,
                instruction=instruction,
                test_result={
                    "git_patch": git_patch,
                },
                metadata=metadata,
                history=history,
                metrics={},
                error=None,
            )

            write_output_to_file(
                instance, process_instance, result, evaluation_state["output_file"]
            )

            logger.info(f"Completed processing instance {instance.instance_id}")

        except Exception as e:
            logger.error(f"Error processing instance {instance.instance_id}: {e}")
            raise  # Re-raise to let the worker handle it

        finally:
            # Clean up conversation
            if conversation:
                conversation.close()

    def get_instance_docker_image(
        instance: pd.Series,
    ) -> str:
        # Official SWE-Bench image
        # swebench/sweb.eval.x86_64.django_1776_django-11333:v1
        # SWE-bench-Live uses the same naming convention as SWE-Bench
        docker_image_prefix = "docker.io/swebench/"
        repo, name = str(instance["instance_id"]).split("__")
        prefix = docker_image_prefix.rstrip("/")
        image_name = f"{prefix}/sweb.eval.x86_64.{repo}_1776_{name}:latest".lower()
        logger.debug(f"Using official SWE-Bench image: {image_name}")
        return image_name

    def complete_dataset_run():
        """Complete the dataset run - any cleanup if needed."""
        logger.info("Remote evaluation completed!")

    # Create and run the Evaluation
    evaluation = Evaluation(
        metadata=metadata,
        initialize_dataset_run=initialize_dataset_run,
        process_instance=process_instance,
        complete_dataset_run=complete_dataset_run,
        num_workers=num_workers,
        get_instance_docker_image=get_instance_docker_image,
    )

    evaluation.run()


def main():
    default_prompt_path = os.path.join(
        os.path.dirname(__file__), "prompts", "default.j2"
    )
    parser = get_parser()
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=default_prompt_path,
        help="Path to prompt template file",
    )
    args = parser.parse_args()

    DATASET = args.dataset
    SPLIT = args.split
    MODEL = args.llm_config
    EVAL_OUTPUT_DIR = args.eval_output_dir
    MAX_ITERATIONS = args.max_iterations
    EVAL_N_LIMIT = args.eval_n_limit
    EVAL_NOTE = args.eval_note
    PROMPT_PATH = args.prompt_path

    # Create LLM instance
    api_key = os.getenv("LITELLM_API_KEY")
    if not api_key:
        raise ValueError("LITELLM_API_KEY environment variable is not set")
    llm = LLM(
        model=MODEL,
        api_key=SecretStr(api_key),
        base_url="https://llm-proxy.eval.all-hands.dev",
        temperature=0,
        service_id="litellm_proxy",
    )

    dataset_description = DATASET.replace("/", "__") + "-" + SPLIT.replace("/", "__")

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
        details={},
        dataset=DATASET,
        data_split=SPLIT,
        prompt_path=PROMPT_PATH,
        eval_n_limit=EVAL_N_LIMIT,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
    )

    # Always use remote evaluation
    run_evaluation(llm, metadata, args.eval_num_workers)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
