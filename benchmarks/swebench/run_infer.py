import json
import os
from typing import Any, List

from jinja2 import Environment, FileSystemLoader

from benchmarks.swebench import constants
from benchmarks.swebench.build_images import (
    extract_custom_tag,
    get_official_docker_image,
    should_wrap_instance_id,
    wrap_image,
)
from benchmarks.swebench.config import INFER_DEFAULTS
from benchmarks.utils.acp import (
    add_acp_agent_metadata,
    build_acp_agent,
    get_acp_forward_env,
    is_acp_agent,
    setup_acp_workspace,
    workspace_keepalive,
)
from benchmarks.utils.args_parser import add_prompt_path_argument, get_parser
from benchmarks.utils.build_utils import ensure_local_image
from benchmarks.utils.console_logging import summarize_instance
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
from benchmarks.utils.image_utils import remote_image_exists
from benchmarks.utils.llm_config import load_llm_config
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
    ToolPresetType,
)
from benchmarks.utils.version import IMAGE_TAG_PREFIX
from openhands.sdk import Agent, Conversation, Tool, get_logger
from openhands.sdk.agent import ACPAgent
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.delegate import DelegateTool
from openhands.workspace import APIRemoteWorkspace, DockerWorkspace


logger = get_logger(__name__)


def get_tools_for_preset(
    preset: ToolPresetType, enable_browser: bool = False
) -> list[Tool]:
    """Get the list of tools for the given preset.

    Args:
        preset: The tool preset to use (default, gemini, or planning).
        enable_browser: Whether to include browser tools.

    Returns:
        List of Tool instances for the given preset.
    """
    if preset == "gemini":
        from openhands.tools.preset.gemini import get_gemini_tools

        return get_gemini_tools(enable_browser=enable_browser)
    elif preset == "planning":
        from openhands.tools.preset.planning import get_planning_tools

        # Planning preset doesn't support browser tools
        return get_planning_tools()
    else:  # default
        from openhands.tools.preset.default import get_default_tools

        return get_default_tools(enable_browser=enable_browser)


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

        For ACPAgent, the provider API key is forwarded to the container.

        Args:
            instance: The evaluation instance to prepare workspace for.
            resource_factor: Resource factor for runtime allocation (default: 1).
                           Higher values allocate more CPU/memory resources.
                           Used by APIRemoteWorkspace for remote runtime allocation.
        """
        forward_env = get_acp_forward_env(self.metadata.agent_type, forward_env)

        official_docker_image = get_official_docker_image(instance.id)
        build_target = constants.DEFAULT_BUILD_TARGET
        custom_tag = extract_custom_tag(official_docker_image)
        # For non-binary targets, append target suffix
        suffix = (
            f"-{build_target}" if build_target != constants.BUILD_TARGET_BINARY else ""
        )
        base_agent_image = (
            f"{EVAL_AGENT_SERVER_IMAGE}:{IMAGE_TAG_PREFIX}-{custom_tag}{suffix}"
        )
        wrap_needed = should_wrap_instance_id(instance.id)
        agent_server_image = base_agent_image

        if self.metadata.workspace_type == "docker":
            built = ensure_local_image(
                agent_server_image=base_agent_image,
                base_image=official_docker_image,
                custom_tag=custom_tag,
                target=build_target,
            )
            if built and wrap_needed:
                wrapped_result = wrap_image(base_agent_image, push=False)
                if wrapped_result.error:
                    raise RuntimeError(
                        "Wrapped image build failed: "
                        f"{wrapped_result.error}; log={wrapped_result.log_path}"
                    )
            elif not built and wrap_needed:
                logger.info(
                    f"Using pre-built image {base_agent_image} "
                    "(assumed already wrapped)"
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
            if not remote_image_exists(agent_server_image):
                raise RuntimeError(
                    f"Agent server image {agent_server_image} does not exist in container registry, "
                    "make sure to build, push it, and make it public accessible before using remote workspace."
                )
            logger.info(
                f"Using remote workspace with image {agent_server_image} "
                f"(tag prefix: {IMAGE_TAG_PREFIX}, resource_factor: {resource_factor})"
            )
            startup_timeout = float(
                os.getenv(
                    "REMOTE_RUNTIME_STARTUP_TIMEOUT",
                    str(constants.DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT),
                )
            )
            workspace = APIRemoteWorkspace(
                runtime_api_url=os.getenv(
                    "RUNTIME_API_URL", constants.DEFAULT_RUNTIME_API_URL
                ),
                runtime_api_key=runtime_api_key,
                server_image=agent_server_image,
                target_type="source" if "source" in build_target else "binary",
                forward_env=forward_env or [],
                resource_factor=resource_factor,
                init_timeout=startup_timeout,
                startup_wait_timeout=startup_timeout,
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
        if is_acp_agent(self.metadata.agent_type):
            agent = build_acp_agent(self.metadata.agent_type, self.metadata.llm.model)
        else:
            tools = get_tools_for_preset(
                preset=self.metadata.tool_preset,
                # Disable browser tools in CLI mode
                enable_browser=False,
            )
            if self.metadata.enable_delegation:
                tools.append(Tool(name=DelegateTool.name))
            condenser = None
            if self.metadata.enable_condenser:
                condenser = LLMSummarizingCondenser(
                    llm=self.metadata.llm.model_copy(update={"usage_id": "condenser"}),
                    max_size=self.metadata.condenser_max_size,
                    keep_first=self.metadata.condenser_keep_first,
                )
            agent = Agent(
                llm=self.metadata.llm,
                tools=tools,
                system_prompt_kwargs={"cli_mode": True},
                condenser=condenser,
                # TODO: we can enable security analyzer later
                # security_analyzer=LLMSecurityAnalyzer(),
            )

        assert isinstance(workspace, RemoteWorkspace)

        setup_acp_workspace(self.metadata.agent_type, workspace)

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
        cp_testebed_repo = workspace.execute_command(
            (f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}")
        )
        assert cp_testebed_repo.exit_code == 0, (
            f"cp_testebed_repo failed: {cp_testebed_repo.stderr}"
        )

        # git reset
        git_reset = workspace.execute_command(f"cd {repo_path} ; git reset --hard")
        assert git_reset.exit_code == 0, f"git reset failed: {git_reset.stderr}"

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path=workspace.working_dir,
        )
        with workspace_keepalive(self.metadata.agent_type, workspace):
            conversation.send_message(instruction)
            # Run conversation with fake user responses to handle agent messages
            run_conversation_with_fake_user_response(conversation)

        # git add
        workspace.execute_command(f"cd {repo_path} ; git add -A")

        # git commit
        # Use --no-verify to bypass pre-commit hooks (e.g., husky) that can fail
        workspace.execute_command(
            f"cd {repo_path} && "
            f"git config --global user.email '{constants.GIT_USER_EMAIL}' && "
            f"git config --global user.name '{constants.GIT_USER_NAME}' && "
            f"git commit --no-verify -m '{constants.GIT_COMMIT_MESSAGE}'"
        )

        # Get git patch
        base_commit = instance.data["base_commit"]
        git_patch_result = workspace.execute_command(
            (f"cd {repo_path} ; git --no-pager diff --no-color {base_commit} HEAD")
        )
        assert git_patch_result.exit_code == 0, (
            f"git diff failed: {git_patch_result.stderr}"
        )
        git_patch = git_patch_result.stdout

        # Log instance summary
        summarize_instance(
            instance_id=instance.id,
            conversation=conversation,
            git_patch=git_patch,
            logger=logger,
        )

        # Build test_result with git patch and optional ACP agent metadata
        test_result: dict[str, Any] = {
            "git_patch": git_patch,
        }
        if isinstance(agent, ACPAgent):
            add_acp_agent_metadata(test_result, agent)

        # EvalOutput is your model; keep fields consistent with prior JSONL
        out = EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result=test_result,
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )
        return out


def main() -> None:
    parser = get_parser()
    add_prompt_path_argument(parser, __file__)
    parser.set_defaults(**INFER_DEFAULTS)
    args = parser.parse_args()

    # Validate n_critic_runs
    if args.n_critic_runs < 1:
        raise ValueError(f"n_critic_runs must be >= 1, got {args.n_critic_runs}")

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
    logger.info(f"Using tool preset: {args.tool_preset}")

    # Handle condenser configuration
    # --disable-condenser takes precedence over --enable-condenser and defaults
    enable_condenser = args.enable_condenser
    if args.disable_condenser:
        enable_condenser = False

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={},
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
        n_critic_runs=args.n_critic_runs,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        tool_preset=args.tool_preset,
        enable_delegation=args.enable_delegation,
        agent_type=args.agent_type,
        enable_condenser=enable_condenser,
        condenser_max_size=args.condenser_max_size,
        condenser_keep_first=args.condenser_keep_first,
    )

    # Run orchestrator with a simple JSONL writer
    evaluator = SWEBenchEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")
    # Emit machine-readable path for callers
    print(json.dumps({"output_json": str(evaluator.output_path)}))


if __name__ == "__main__":
    main()
