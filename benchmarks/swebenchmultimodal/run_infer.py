import json
import os
from pathlib import Path
from typing import List

import requests
from jinja2 import Environment, FileSystemLoader

from benchmarks.swebenchmultimodal.build_images import (
    extract_custom_tag,
    get_official_docker_image,
)
from benchmarks.swebenchmultimodal.config import INFER_DEFAULTS
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.build_utils import build_image
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.conversation import build_event_persistence_callback
from benchmarks.utils.critics import create_critic
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response
from benchmarks.utils.llm_config import load_llm_config
from benchmarks.utils.image_utils import image_exists
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
)
from benchmarks.utils.version import IMAGE_TAG_PREFIX
from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    ImageContent,
    Message,
    TextContent,
    get_logger,
)
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import APIRemoteWorkspace, DockerWorkspace


logger = get_logger(__name__)


def is_valid_image_url(url: str, allowed_types: list | None = None) -> bool:
    """
    Check if a URL points to a valid image by examining the HTTP response content type.

    Args:
        url: The URL to check
        allowed_types: List of allowed MIME types. If None, defaults to common image types.

    Returns:
        True if URL points to a valid image type, False otherwise
    """
    if allowed_types is None:
        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]

    try:
        # Send a HEAD request first to check headers without downloading the entire file
        response = requests.head(url, allow_redirects=True, timeout=5)
        response.raise_for_status()

        # Get the content type from the response headers
        content_type = response.headers.get("Content-Type", "")

        # Check if the content type is in the allowed types
        return any(content_type.startswith(t) for t in allowed_types)
    except Exception:
        return False


def get_instruction(
    instance: dict,
    metadata: EvalMetadata,
    workspace_path: str,
) -> str:
    """Generate instruction for the agent."""
    workspace_dir_name = instance["repo"].split("/")[-1]
    assert metadata.details is not None

    # Set up Jinja2 environment
    assert metadata.prompt_path is not None
    prompts_dir = os.path.dirname(metadata.prompt_path)
    template_name = os.path.basename(metadata.prompt_path)
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


class SWEBenchEvaluation(Evaluation):
    """
    Process-based SWE-bench evaluation implemented as a child of the
    abstract Evaluation orchestrator.

    Implements:
      - prepare_instances()
      - prepare_workspace(instance)
      - evaluate_instance(instance, workspace)
    """

    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up SWE-bench evaluation data")

        df = get_dataset(
            dataset_name=self.metadata.dataset,
            split=self.metadata.dataset_split,
            eval_limit=self.metadata.eval_limit,
            selected_instances_file=self.metadata.selected_instances_file,
        )

        instances: List[EvalInstance] = []
        for _, row in df.iterrows():
            inst_id = str(row["instance_id"])
            instances.append(EvalInstance(id=inst_id, data=row.to_dict()))

        logger.info("Total instances to process: %d", len(instances))
        return instances

    # ---- Hook: prepare a workspace per instance ----------------------------------
    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """
        Use DockerWorkspace by default.

        Args:
            instance: The evaluation instance to prepare workspace for.
            resource_factor: Resource factor for runtime allocation (default: 1).
                           Higher values allocate more CPU/memory resources.
                           Used by APIRemoteWorkspace for remote runtime allocation.
            forward_env: Environment variables to forward into the workspace.
        """
        # Use multimodal image
        official_docker_image = get_official_docker_image(instance.id)
        build_target = "source-minimal"
        custom_tag = extract_custom_tag(official_docker_image)
        # For non-binary targets, append target suffix
        suffix = f"-{build_target}" if build_target != "binary" else ""

        if self.metadata.workspace_type == "docker":
            agent_server_image = (
                f"{EVAL_AGENT_SERVER_IMAGE}:{IMAGE_TAG_PREFIX}-{custom_tag}{suffix}"
            )
            SKIP_BUILD = os.getenv("SKIP_BUILD", "1").lower() in ("1", "true", "yes")
            logger.info(f"SKIP_BUILD={SKIP_BUILD}")
            if not SKIP_BUILD:
                logger.info(
                    f"Building workspace from {official_docker_image} "
                    f"for instance {instance.id}. "
                    "This may take a while...\n"
                    "You can run benchmarks/swebenchmultimodal/build_images.py and set "
                    "SWE_BENCH_SKIP_BUILD=1 to skip building and use pre-built "
                    "agent-server image."
                )

                output = build_image(
                    base_image=official_docker_image,
                    target_image=EVAL_AGENT_SERVER_IMAGE,
                    custom_tag=custom_tag,
                    target=build_target,
                    push=False,
                )
                logger.info(f"Image build output: {output}")
                assert output.error is None, f"Image build failed: {output.error}"
                if agent_server_image not in output.tags:
                    raise RuntimeError(
                        f"Built image tags {output.tags} do not include expected tag "
                        f"{agent_server_image}"
                    )

            workspace = DockerWorkspace(
                server_image=agent_server_image,
                working_dir="/workspace",
                forward_env=forward_env or [],
            )
        elif self.metadata.workspace_type == "remote":
            runtime_api_key = os.getenv("RUNTIME_API_KEY")
            if not runtime_api_key:
                raise ValueError(
                    "RUNTIME_API_KEY environment variable is not set for remote workspace"
                )

            agent_server_image = (
                f"{EVAL_AGENT_SERVER_IMAGE}:{IMAGE_TAG_PREFIX}-{custom_tag}{suffix}"
            )
            if not image_exists(agent_server_image):
                raise RuntimeError(
                    f"Agent server image {agent_server_image} does not exist in container registry, "
                    "make sure to build, push it, and make it public accessible before using remote workspace."
                )
            logger.info(
                f"Using remote workspace with image {agent_server_image} "
                f"(tag prefix: {IMAGE_TAG_PREFIX}, resource_factor: {resource_factor})"
            )
            startup_timeout = float(os.getenv("REMOTE_RUNTIME_STARTUP_TIMEOUT", "600"))
            workspace = APIRemoteWorkspace(
                runtime_api_url=os.getenv(
                    "RUNTIME_API_URL", "https://runtime.eval.all-hands.dev"
                ),
                runtime_api_key=runtime_api_key,
                server_image=agent_server_image,
                init_timeout=startup_timeout,
                startup_wait_timeout=startup_timeout,
                target_type="source" if "source" in build_target else "binary",
                forward_env=forward_env or [],
                resource_factor=resource_factor,
            )
        else:
            raise ValueError(
                f"Unsupported workspace_type: {self.metadata.workspace_type}"
            )

        for cmd in self.metadata.env_setup_commands or []:
            res = workspace.execute_command(cmd)
            if res.exit_code != 0:
                raise RuntimeError(
                    f"Failed to run env setup command '{cmd}': {res.stderr}"
                )
            logger.debug(f"Ran env setup command '{cmd}': {res.stdout}")
        return workspace

    # ---- Hook: evaluate one instance ---------------------------------------------
    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """
        Create conversation, run agent, collect history and git patch.
        Do not write files here; just return EvalOutput.
        """
        tools = get_default_tools(
            # Enable browser tools for frontend development tasks
            enable_browser=True,
        )
        agent = Agent(
            llm=self.metadata.llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
            # TODO: we can enable condenser and security analyzer later
            # and have them configurable via EvalMetadata
            # condenser=get_default_condenser(
            #     llm=self.metadata.llm.model_copy(update={"service_id": "condenser"})
            # ),
            # security_analyzer=LLMSecurityAnalyzer(),
        )

        assert isinstance(workspace, RemoteWorkspace)

        repo_path = f"/workspace/{instance.data['repo'].split('/')[-1]}/"
        instance.data["repo_path"] = repo_path

        persist_callback = build_event_persistence_callback(
            run_id=self.metadata.eval_output_dir,
            instance_id=instance.id,
            attempt=self.current_attempt,
        )

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[persist_callback],
            max_iteration_per_run=self.metadata.max_iterations,
            delete_on_close=True,
        )

        logger.info("repo_path: %s", repo_path)
        # Copy testbed repo to workspace (same as regular swebench)
        # The multimodal benchmark uses regular SWE-bench images which have /testbed
        cp_testbed_repo = workspace.execute_command(
            f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}"
        )
        assert cp_testbed_repo.exit_code == 0, (
            f"cp_testbed_repo failed: {cp_testbed_repo.stderr}"
        )

        # git reset to clean state
        git_reset = workspace.execute_command(f"cd {repo_path} ; git reset --hard")
        assert git_reset.exit_code == 0, f"git reset failed: {git_reset.stderr}"

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path=workspace.working_dir,
        )

        # Handle image assets for multimodal instances
        if "image_assets" in instance.data and instance.data["image_assets"]:
            try:
                assets = json.loads(instance.data["image_assets"])
                if "problem_statement" in assets and assets["problem_statement"]:
                    image_urls = assets["problem_statement"]

                    # Filter and validate image URLs
                    valid_urls = []
                    index_dict = {}
                    for url in image_urls:
                        if is_valid_image_url(url):
                            if url in instruction:
                                valid_urls.append(url)
                                idx = instruction.find(url)
                                index_dict[url] = idx
                            else:
                                logger.warning(
                                    f"Image URL {url} not found in instruction, skipping"
                                )
                        else:
                            logger.info(
                                f"Image URL {url} is invalid or inaccessible, skipping"
                            )

                    if valid_urls:
                        # Sort URLs by their position in the instruction
                        sorted_urls = sorted(index_dict.items(), key=lambda x: x[1])
                        sorted_urls = [item[0] for item in sorted_urls]

                        # Add image numbering to instruction
                        modified_instruction = instruction
                        for idx, url in enumerate(sorted_urls):
                            modified_instruction = modified_instruction.replace(
                                url, f"{url} (Image: {idx + 1})"
                            )

                        logger.info(
                            f"Sending instruction with {len(sorted_urls)} valid images"
                        )

                        # Create message with both text and images
                        message = Message(
                            role="user",
                            content=[
                                TextContent(text=modified_instruction),
                                ImageContent(image_urls=sorted_urls),
                            ],
                        )
                        conversation.send_message(message)
                    else:
                        logger.info(
                            "No valid image URLs found, sending text-only instruction"
                        )
                        conversation.send_message(instruction)
                else:
                    logger.info("No problem_statement images found in image_assets")
                    conversation.send_message(instruction)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse image_assets: {e}")
                conversation.send_message(instruction)
        else:
            logger.info("No image_assets found, sending text-only instruction")
            conversation.send_message(instruction)
        # Run conversation with fake user responses to handle agent messages
        run_conversation_with_fake_user_response(
            conversation, run_timeout=self.metadata.conversation_timeout
        )

        # git add
        workspace.execute_command(f"cd {repo_path} ; git add -A")

        # git commit (same as regular swebench - includes git config)
        # Use --no-verify to bypass pre-commit hooks (e.g., husky) that can fail
        # and prevent the commit from being created
        workspace.execute_command(
            f"cd {repo_path} && "
            "git config --global user.email 'evaluation@openhands.dev' && "
            "git config --global user.name 'OpenHands Evaluation' && "
            "git commit --no-verify -m 'patch'"
        )

        # Get git patch (same as regular swebench - use base_commit)
        base_commit = instance.data["base_commit"]
        git_patch_result = workspace.execute_command(
            f"cd {repo_path} ; git --no-pager diff --no-color {base_commit} HEAD"
        )
        assert git_patch_result.exit_code == 0, (
            f"git diff failed: {git_patch_result.stderr}"
        )
        git_patch = git_patch_result.stdout

        # EvalOutput is your model; keep fields consistent with prior JSONL
        out = EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result={
                "git_patch": git_patch,
            },
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )
        return out


def main() -> None:
    prompt_dir = (Path(__file__).parent / "prompts").resolve()
    try:
        choices = [str(p.relative_to(Path.cwd())) for p in prompt_dir.glob("*.j2")]
    except ValueError:
        choices = [str(p) for p in prompt_dir.glob("*.j2")]
    default_prompt_path = prompt_dir / "default.j2"
    assert default_prompt_path.exists(), (
        f"Default prompt {default_prompt_path} not found"
    )

    parser = get_parser()
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=str(default_prompt_path),
        choices=choices,
        help="Path to prompt template file",
    )
    # Apply INFER_DEFAULTS from config (matches evaluation repository values.yaml)
    parser.set_defaults(**INFER_DEFAULTS)
    args = parser.parse_args()

    # Validate max_attempts
    if args.max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {args.max_attempts}")

    llm = load_llm_config(args.llm_config_path)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    dataset_description = (
        args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__")
    )

    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )

    # Create critic instance from parsed arguments
    critic = create_critic(args)
    logger.info(f"Using critic: {type(critic).__name__}")

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        conversation_timeout=args.conversation_timeout,
        eval_output_dir=structured_output_dir,
        details={},
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
        max_attempts=args.max_attempts,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        skip_failed_samples=args.skip_failed_samples,
        workspace_type=args.workspace,
    )

    # Run orchestrator with a simple JSONL writer
    evaluator = SWEBenchEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")
    print(json.dumps({"output_json": str(evaluator.output_path)}))


if __name__ == "__main__":
    main()
