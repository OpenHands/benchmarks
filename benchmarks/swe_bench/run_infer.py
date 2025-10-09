from __future__ import annotations

import os
from typing import Any

from pydantic import SecretStr

from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.conversation_tools import get_git_patch_from_history, get_history
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_instruction,
    make_metadata,
    read_completed_instances,
    write_output_to_file,
)
from benchmarks.utils.instance import Instance
from benchmarks.utils.shared import EvalMetadata
from openhands.sdk import LLM, Agent, get_logger
from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation
from openhands.tools.preset.default import get_default_tools


logger = get_logger(__name__)


class SWEBenchEvaluation(Evaluation):
    """SWE-bench specific evaluation implementation."""

    def __init__(self, metadata: EvalMetadata, num_workers: int = 1):
        super().__init__(metadata, num_workers)
        self.llm_instance = None
        self.agent = None
        self.instances = None
        self.output_file = None
        self.results = []
        self.workspace_path = "/workspace"

    def setup_data(self):
        """Initialize dataset and prepare for evaluation."""
        logger.info("Setting up SWE-bench evaluation data")

        # Set up output file
        self.output_file = os.path.join(self.metadata.eval_output_dir, "output.jsonl")

        # Get dataset
        dataset = get_dataset(
            self.metadata.dataset,
            self.metadata.data_split,
            self.output_file,
            self.metadata.eval_n_limit,
        )

        # Filter dataset if needed
        if self.metadata.max_iterations is not None:
            dataset = dataset.head(self.metadata.max_iterations)

        # Convert to Instance objects
        self.instances = []
        for _, row in dataset.iterrows():
            instance = Instance(id=row["instance_id"], data=row.to_dict())
            self.instances.append(instance)

        # Read completed instances to avoid re-processing
        completed_instances = read_completed_instances(self.output_file)
        self.instances = [
            instance
            for instance in self.instances
            if instance.id not in completed_instances
        ]

        logger.info(f"Total instances to process: {len(self.instances)}")
        return self.instances

    def before_eval(self):
        """Setup before evaluation starts."""
        logger.info("Starting SWE-bench evaluation")

        # Initialize agent
        self.agent = Agent(
            llm=self.llm_instance,
            tools=get_default_tools(),
        )

    def process_instance(self, instance: Instance, workspace):
        """Process a single instance."""
        logger.info(f"Processing instance {instance.id}")

        # Get instance data
        instance_data = instance.data

        # Create conversation
        conversation = RemoteConversation(
            workspace.server_url,
            workspace.workspace_id,
            SecretStr(workspace.token),
        )

        # Get instruction
        instruction = get_instruction(instance_data, self.metadata.agent_class)

        def event_callback(event) -> None:
            """Handle events during conversation."""
            pass

        # Run conversation
        try:
            conversation.add_message(
                role="user",
                content=instruction,
            )

            # Set up event logging
            def log_event(event):
                logger.debug(f"Event: {event}")

            # Run agent
            self.agent.run(
                conversation=conversation,
                event_callback=log_event,
                max_iterations=self.metadata.max_iterations,
            )

            # Get history and patch
            history = get_history(conversation)
            git_patch = get_git_patch_from_history(history)

            # Prepare result
            result = {
                "instance_id": instance.id,
                "instruction": instruction,
                "git_patch": git_patch,
                "history": history,
                "test_result": {},
            }

            # Write result
            write_output_to_file(result, self.output_file)
            self.results.append(result)

            logger.info(f"Completed instance {instance.id}")

        except Exception as e:
            logger.error(f"Error processing instance {instance.id}: {e}")
            # Write error result
            error_result = {
                "instance_id": instance.id,
                "instruction": instruction,
                "git_patch": "",
                "history": [],
                "test_result": {"error": str(e)},
            }
            write_output_to_file(error_result, self.output_file)

    def get_instance_docker_image(self, instance: Instance) -> str:
        """Get Docker image for the instance."""
        instance_data = instance.data

        # Default image
        image_name = "python:3.11-bookworm"

        # Get repo-specific image if available
        if "repo" in instance_data:
            repo = instance_data["repo"]
            # Map repo to specific image if needed
            repo_images = {
                # Add specific repo mappings here if needed
            }
            if repo in repo_images:
                image_name = repo_images[repo]

        return image_name

    def on_evaluation_completion(self):
        """Handle completion of evaluation."""
        logger.info("SWE-bench evaluation completed")
        logger.info(f"Processed {len(self.results)} instances")
        logger.info(f"Results written to {self.output_file}")


def run_evaluation(llm: Any, metadata: EvalMetadata, num_workers: int = 1):
    """
    Run evaluation using agent server.

    Args:
        llm: LLM instance to use for evaluation
        metadata: EvalMetadata object containing evaluation configuration
        num_workers: Number of worker threads to use for parallel processing
    """
    # Create SWEBenchEvaluation instance
    evaluation = SWEBenchEvaluation(metadata, num_workers)
    evaluation.llm_instance = llm

    # Run the evaluation
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
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY environment variable is not set")
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
