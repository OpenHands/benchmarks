import fcntl
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import List, Sequence, cast

import huggingface_hub
import pandas as pd
from datasets import DatasetDict, load_dataset
from PIL import Image

from benchmarks.gaia.scorer import question_scorer
from benchmarks.gaia.utils import image_to_jpg_base64_url, image_to_png_base64_url
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import construct_eval_output_dir
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    ImageContent,
    Message,
    MessageEvent,
    TextContent,
    get_logger,
)
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerWorkspace


logger = get_logger(__name__)

# Cache directory for GAIA dataset files
DATASET_CACHE_DIR = Path(__file__).parent / "data"


class GAIAEvaluation(Evaluation):
    """
    GAIA benchmark evaluation implemented as a child of the
    abstract Evaluation orchestrator.

    Implements:
      - prepare_instances()
      - prepare_workspace(instance)
      - evaluate_instance(instance, workspace)
    """

    def prepare_instances(self) -> List[EvalInstance]:
        """Load GAIA dataset from HuggingFace and prepare instances."""
        logger.info("Setting up GAIA evaluation data")

        # Load dataset from HuggingFace
        assert self.metadata.details is not None
        level = self.metadata.details.get("level")
        if not level:
            raise ValueError(
                "GAIA level must be specified in metadata.details['level']"
            )

        logger.info(
            f"Loading GAIA dataset: {level}, split: {self.metadata.dataset_split}"
        )
        dataset = cast(DatasetDict, load_dataset("gaia-benchmark/GAIA", level))

        # Download dataset files
        logger.info(f"Downloading GAIA dataset files to {DATASET_CACHE_DIR}")
        DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        huggingface_hub.snapshot_download(
            "gaia-benchmark/GAIA",
            repo_type="dataset",
            local_dir=str(DATASET_CACHE_DIR),
        )

        # Convert to pandas and rename task_id to instance_id
        df = cast(pd.DataFrame, dataset[self.metadata.dataset_split].to_pandas())
        df.rename(columns={"task_id": "instance_id"}, inplace=True)

        # Filter completed instances
        completed_instances = self._get_completed_instances()
        if completed_instances:
            df = cast(
                pd.DataFrame, df[~df["instance_id"].isin(list(completed_instances))]
            )
            logger.info(f"Filtered out {len(completed_instances)} completed instances")

        # Apply eval_limit if specified
        if self.metadata.eval_limit and self.metadata.eval_limit > 0:
            df = cast(pd.DataFrame, df.head(self.metadata.eval_limit))
            logger.info(f"Limited to {len(df)} instances due to eval_limit")

        # Filter by selected_instances_file if provided
        if self.metadata.selected_instances_file:
            with open(self.metadata.selected_instances_file, "r") as f:
                selected_ids = set(line.strip() for line in f if line.strip())
            df = cast(pd.DataFrame, df[df["instance_id"].isin(list(selected_ids))])
            logger.info(f"Filtered to {len(df)} selected instances")

        instances: List[EvalInstance] = []
        for _, row in df.iterrows():
            inst_id = str(row["instance_id"])
            instances.append(EvalInstance(id=inst_id, data=row.to_dict()))

        logger.info(f"Total instances to process: {len(instances)}")
        return instances

    def prepare_workspace(self, instance: EvalInstance) -> RemoteWorkspace:
        """Create workspace and copy necessary files."""
        logger.info(f"Preparing workspace for instance {instance.id}")

        # Create Docker workspace with python-nodejs image
        workspace = DockerWorkspace(
            base_image="nikolaik/python-nodejs:python3.12-nodejs22",
            working_dir="/workspace",
        )

        # Create workspace directory
        workspace.execute_command("mkdir -p /workspace")

        # Handle file if present
        file_name = instance.data.get("file_name", "")
        if file_name:
            logger.info(f"Handling file: {file_name}")
            assert self.metadata.details is not None

            # Construct source file path
            src_file = (
                DATASET_CACHE_DIR / "2023" / self.metadata.dataset_split / file_name
            )

            if not src_file.exists():
                logger.warning(f"Source file not found: {src_file}")
            else:
                extension_name = file_name.split(".")[-1].lower()

                # Skip images (jpg, png) - they'll be passed as base64 URLs
                if extension_name in ["jpg", "png", "jpeg"]:
                    logger.info(
                        f"Skipping image file {file_name} (will be passed as URL)"
                    )
                elif extension_name == "zip":
                    # Extract zip files
                    logger.info(f"Extracting zip file {file_name}")
                    with tempfile.TemporaryDirectory() as temp_dir:
                        with zipfile.ZipFile(src_file, "r") as zip_ref:
                            zip_ref.extractall(temp_dir)
                        # Copy all extracted files to workspace
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                local_path = os.path.join(root, file)
                                workspace.file_upload(local_path, f"/workspace/{file}")
                else:
                    # Copy other files
                    logger.info(f"Copying file {file_name} to workspace")
                    workspace.file_download(
                        str(src_file), f"/workspace/file.{extension_name}"
                    )

        # Install ffmpeg (some GAIA tasks need it)
        logger.info("Installing ffmpeg...")
        result = workspace.execute_command(
            "apt-get update && apt-get install -y ffmpeg ffprobe"
        )
        if result.exit_code != 0:
            logger.warning(f"Failed to install ffmpeg: {result.stderr}")

        return workspace

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """
        Run agent on a single GAIA instance and evaluate the result.
        """
        logger.info(f"Evaluating instance {instance.id}")

        # Build instruction
        instruction = self._build_instruction(instance)

        # Handle image URLs if the file is an image
        image_urls = []
        file_name = instance.data.get("file_name", "")
        if file_name:
            extension_name = file_name.split(".")[-1].lower()
            if extension_name in ["jpg", "png", "jpeg"]:
                # Load image and encode as base64
                assert self.metadata.details is not None
                src_file = (
                    DATASET_CACHE_DIR / "2023" / self.metadata.dataset_split / file_name
                )
                if src_file.exists():
                    image = Image.open(src_file)
                    if extension_name in ["jpg", "jpeg"]:
                        image_urls.append(image_to_jpg_base64_url(image))
                    else:
                        image_urls.append(image_to_png_base64_url(image))

        # Create agent
        tools = get_default_tools(enable_browser=True)
        agent = Agent(
            llm=self.metadata.llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
            mcp_config={
                "mcpServers": {
                    "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}
                }
            },
        )

        # Create conversation
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[lambda ev: logger.debug("Event: %s", ev)],
            max_iteration_per_run=self.metadata.max_iterations,
        )

        # Send message and run
        if image_urls:
            msg = Message(
                role="user",
                content=[
                    TextContent(text=instruction),
                    ImageContent(image_urls=image_urls),
                ],
            )
            conversation.send_message(msg)
        else:
            conversation.send_message(instruction)
        conversation.run()

        # Extract answer from conversation history
        model_answer_raw = self._extract_answer_from_history(conversation.state.events)
        model_answer = self._parse_solution_tag(model_answer_raw)

        # Score the answer
        ground_truth = instance.data.get("Final answer", "")
        score = question_scorer(model_answer, ground_truth)

        logger.info(
            f"Instance {instance.id}: score={score}, "
            f"model_answer='{model_answer}', ground_truth='{ground_truth}'"
        )

        # Collect history
        history = list(map(lambda event: event.model_dump(), conversation.state.events))

        # Return evaluation output
        return EvalOutput(
            instance_id=instance.id,
            test_result={
                "score": score,
                "model_answer_raw": model_answer_raw,
                "model_answer": model_answer,
                "ground_truth": ground_truth,
            },
            instruction=instruction,
            error=None,
            history=history,
            instance=instance.data,
        )

    def _build_instruction(self, instance: EvalInstance) -> str:
        """Build GAIA-specific instruction for the agent."""
        question = instance.data.get("Question", "")
        file_name = instance.data.get("file_name", "")

        instruction = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist). Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
You must make sure you find the correct answer! You MUST strictly follow the task-specific formatting instructions for your final answer.
Here is the task:
{task_question}
""".format(
            task_question=question,
        )

        # Add file information if present
        if file_name:
            extension_name = file_name.split(".")[-1].lower()
            if extension_name == "zip":
                # List files from zip
                # level = self.metadata.details and self.metadata.details.get("level", "")
                src_file = (
                    DATASET_CACHE_DIR / "2023" / self.metadata.dataset_split / file_name
                )
                if src_file.exists():
                    with zipfile.ZipFile(src_file, "r") as zip_ref:
                        filenames = [f"/workspace/{f}" for f in zip_ref.namelist()]
                    filenames_str = ", ".join(filenames)
                    instruction += f"To solve this task you will have to use the attached files provided in the workspace at locations: {filenames_str}\n\n"
            elif extension_name in ["jpg", "png", "jpeg"]:
                instruction += "Image: To solve this task you will have to use the image shown below.\n\n"
            else:
                instruction += f"To solve this task you will have to use the attached file provided in the workspace at location: /workspace/file.{extension_name}\n\n"

        # Add GAIA-specific instructions
        instruction += """IMPORTANT: When seeking information from a website, REFRAIN from arbitrary URL navigation. You should utilize the designated search engine tool with precise keywords to obtain relevant URLs or use the specific website's search interface. DO NOT navigate directly to specific URLs as they may not exist.

For example: if you want to search for a research paper on Arxiv, either use the search engine tool with specific keywords or navigate to arxiv.org and then use its interface.
"""
        instruction += "IMPORTANT: You should NEVER ask for Human Help.\n"
        instruction += "IMPORTANT: Please encapsulate your final answer (answer ONLY) within <solution> and </solution> and report it back to users via a message, instead of the 'finish' tool. Your answer will be evaluated using string matching approaches so it important that you STRICTLY adhere to the output formatting instructions specified in the task (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)\n"
        instruction += (
            "For example: The answer to the question is <solution> 42 </solution>.\n"
        )
        instruction += "IMPORTANT: Your final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, express it numerically (i.e., with digits rather than words), do not use commas, and do not include units such as $ or percent signs unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities). If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"

        return instruction

    def _extract_answer_from_history(self, events: Sequence[Event]) -> str:
        """Extract the last agent message/thought from conversation history."""
        # Search backwards through events for agent output
        logger.info(f"All conversation events: {events}")
        for event in reversed(events):
            if isinstance(event, MessageEvent) and event.source == "agent":
                logger.info(f"Found agent message event: {event}")
                # Try different event types
                if event.llm_message and event.llm_message.content:
                    content = event.llm_message.content[0]
                    assert isinstance(content, TextContent)
                    return content.text

        logger.warning("Could not find agent output in history")
        return ""

    def _parse_solution_tag(self, text: str) -> str:
        """Parse solution from <solution>...</solution> tags."""
        matches = re.findall(r"<solution>(.*?)</solution>", text, re.DOTALL)
        if matches:
            return matches[-1].strip()  # Return last match
        else:
            logger.warning(f"No <solution> tag found in: {text[:200]}...")
            return text  # Return raw text as fallback


def main() -> None:
    """Main entry point for GAIA evaluation."""
    parser = get_parser()
    parser.add_argument(
        "--level",
        type=str,
        required=True,
        help="GAIA level to evaluate (e.g., 2023_level1, 2023_level2, 2023_level3)",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {args.max_attempts}")

    # Load LLM config
    llm_config_path = args.llm_config_path
    if not os.path.isfile(llm_config_path):
        raise ValueError(f"LLM config file {llm_config_path} does not exist")
    with open(llm_config_path, "r") as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    # Construct dataset description
    dataset_description = f"gaia-{args.level}-{args.split}"

    # Construct output directory
    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )

    # Create metadata
    metadata = EvalMetadata(
        llm=llm,
        dataset="gaia-benchmark/GAIA",
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={"level": args.level},
        eval_limit=args.n_limit,
        max_attempts=args.max_attempts,
        critic_name=args.critic,
        selected_instances_file=args.select,
    )

    # Create evaluator
    evaluator = GAIAEvaluation(metadata=metadata, num_workers=args.num_workers)

    # Define result writer
    def _default_on_result_writer(eval_output_dir: str):
        def _cb(instance: EvalInstance, out: EvalOutput) -> None:
            with open(evaluator.output_path, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(out.model_dump_json() + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)

        return _cb

    # Run evaluation
    evaluator.run(on_result=_default_on_result_writer(metadata.eval_output_dir))

    logger.info("Evaluation completed!")
    logger.info(f"Results written to: {evaluator.output_path}")


if __name__ == "__main__":
    main()
