import json
import os
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader

from benchmarks.swebenchmultimodal.build_images import (
    extract_custom_tag,
    get_official_docker_image,
)
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.build_utils import build_image
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.critics import create_critic
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.image_utils import image_exists
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
)
from benchmarks.utils.version import SDK_SHORT_SHA
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
    def prepare_workspace(self, instance: EvalInstance) -> RemoteWorkspace:
        """
        Use DockerWorkspace by default.
        """
        # Use multimodal image
        official_docker_image = get_official_docker_image(instance.id)
        build_target = "source-minimal"
        custom_tag = extract_custom_tag(official_docker_image)
        # For non-binary targets, append target suffix
        suffix = f"-{build_target}" if build_target != "binary" else ""

        if self.metadata.workspace_type == "docker":
            agent_server_image = (
                f"{EVAL_AGENT_SERVER_IMAGE}:{SDK_SHORT_SHA}-{custom_tag}{suffix}"
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
            )
        elif self.metadata.workspace_type == "remote":
            runtime_api_key = os.getenv("RUNTIME_API_KEY")
            sdk_short_sha = os.getenv("SDK_SHORT_SHA", SDK_SHORT_SHA)
            if not runtime_api_key:
                raise ValueError(
                    "RUNTIME_API_KEY environment variable is not set for remote workspace"
                )

            agent_server_image = (
                f"{EVAL_AGENT_SERVER_IMAGE}:{sdk_short_sha}-{custom_tag}{suffix}"
            )
            if not image_exists(agent_server_image):
                raise RuntimeError(
                    f"Agent server image {agent_server_image} does not exist in container registry, "
                    "make sure to build, push it, and make it public accessible before using remote workspace."
                )
            logger.info(
                f"Using remote workspace with image {agent_server_image} (sdk sha: {sdk_short_sha})"
            )
            workspace = APIRemoteWorkspace(
                runtime_api_url=os.getenv(
                    "RUNTIME_API_URL", "https://runtime.eval.all-hands.dev"
                ),
                runtime_api_key=runtime_api_key,
                server_image=agent_server_image,
                resource_factor=self.metadata.runtime_resource_factor,
                target_type="source" if "source" in build_target else "binary",
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
            # Disable browser tools in CLI mode
            enable_browser=False,
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

        def _log_event(ev):  # keep it simple
            logger.debug("Event: %s", ev)

        repo_path = f"/workspace/{instance.data['repo'].split('/')[-1]}/"
        instance.data["repo_path"] = repo_path

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[_log_event],
            max_iteration_per_run=self.metadata.max_iterations,
        )

        logger.info("repo_path: %s", repo_path)
        # For multimodal datasets, we don't copy from testbed since it doesn't exist
        # The repo should already be available in the workspace
        logger.info("Skipping testbed copy for multimodal dataset")

        # Create repo directory if it doesn't exist
        mkdir_result = workspace.execute_command(f"mkdir -p {repo_path}")
        if mkdir_result.exit_code != 0:
            logger.warning(f"mkdir failed: {mkdir_result.stderr}")

        # Initialize git repo if it doesn't exist, or reset if it does
        git_init_result = workspace.execute_command(f"cd {repo_path} ; git init")
        if git_init_result.exit_code != 0:
            logger.warning(f"git init failed: {git_init_result.stderr}")

        # Configure git user globally in the workspace
        workspace.execute_command(
            "git config --global user.email 'evaluation@openhands.dev' && "
            "git config --global user.name 'OpenHands Evaluation'"
        )

        # Create an empty initial commit to establish HEAD
        initial_commit_result = workspace.execute_command(
            f"cd {repo_path} ; git commit --allow-empty -m 'Initial empty commit'"
        )
        if initial_commit_result.exit_code != 0:
            logger.warning(
                f"Initial empty commit failed: {initial_commit_result.stderr}"
            )

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
                    logger.info(f"Sending instruction with {len(image_urls)} images")

                    # Create message with both text and images
                    message = Message(
                        role="user",
                        content=[
                            TextContent(text=instruction),
                            ImageContent(image_urls=image_urls),
                        ],
                    )
                    conversation.send_message(message)
                else:
                    logger.info("No problem_statement images found in image_assets")
                    conversation.send_message(instruction)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse image_assets: {e}")
                conversation.send_message(instruction)
        else:
            logger.info("No image_assets found, sending text-only instruction")
            conversation.send_message(instruction)
        conversation.run()

        # git add
        workspace.execute_command(f"cd {repo_path} ; git add -A")

        # Check if there are any changes to commit
        status_result = workspace.execute_command(
            f"cd {repo_path} ; git status --porcelain"
        )

        if status_result.exit_code == 0 and status_result.stdout.strip():
            # There are changes to commit
            commit_result = workspace.execute_command(
                f"cd {repo_path} ; git commit -m 'patch'"
            )
            if commit_result.exit_code != 0:
                logger.warning(f"git commit failed: {commit_result.stderr}")
                git_patch = ""
            else:
                # Get git patch - diff between the initial commit and the current commit
                git_patch_result = workspace.execute_command(
                    (f"cd {repo_path} ; git --no-pager diff --no-color HEAD~1 HEAD")
                )
                if git_patch_result.exit_code != 0:
                    logger.warning(f"git diff HEAD~1 failed: {git_patch_result.stderr}")
                    # Try to get the last commit as a patch
                    git_patch_result = workspace.execute_command(
                        (f"cd {repo_path} ; git --no-pager show --no-color HEAD")
                    )
                    if git_patch_result.exit_code != 0:
                        logger.warning("git show HEAD also failed, using empty patch")
                        git_patch = ""
                    else:
                        git_patch = git_patch_result.stdout
                else:
                    git_patch = git_patch_result.stdout
        else:
            # No changes to commit
            logger.warning("No changes detected, using empty patch")
            git_patch = ""

        # EvalOutput is your model; keep fields consistent with prior JSONL
        out = EvalOutput(
            instance_id=instance.id,
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
    choices = [str(p.relative_to(Path.cwd())) for p in prompt_dir.glob("*.j2")]
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
    # Override the default dataset and split for multimodal
    parser.set_defaults(dataset="princeton-nlp/SWE-bench_Multimodal", split="dev")
    args = parser.parse_args()

    # Validate max_attempts
    if args.max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {args.max_attempts}")

    llm_config_path = args.llm_config_path
    if not os.path.isfile(llm_config_path):
        raise ValueError(f"LLM config file {llm_config_path} does not exist")
    with open(llm_config_path, "r") as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)
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
        eval_output_dir=structured_output_dir,
        details={"runtime_resource_factor": args.runtime_resource_factor},
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
        max_attempts=args.max_attempts,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        runtime_resource_factor=args.runtime_resource_factor,
    )

    # Run orchestrator with a simple JSONL writer
    evaluator = SWEBenchEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
