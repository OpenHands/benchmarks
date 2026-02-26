import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, List

from jinja2 import Environment, FileSystemLoader
from pydantic import Field

from benchmarks.swefficiency import constants
from benchmarks.swefficiency.config import DOCKER_DEFAULTS, INFER_DEFAULTS
from benchmarks.swefficiency.workspace import ResourceLimitedDockerWorkspace
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.build_utils import build_image
from benchmarks.utils.conversation import build_event_persistence_callback
from benchmarks.utils.critics import create_critic
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response
from benchmarks.utils.image_utils import image_exists
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
)
from benchmarks.utils.version import SDK_SHORT_SHA
from openhands.sdk import LLM, Agent, Conversation, get_logger
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import APIRemoteWorkspace


logger = get_logger(__name__)

# Agent server image for evaluation
EVAL_AGENT_SERVER_IMAGE = "ghcr.io/openhands/eval-agent-server"


def get_swefficiency_workspace_dir_name(instance: dict) -> str:
    """Get the workspace directory name for a SWE-fficiency instance."""
    repo = instance["repo"].replace("/", "__")
    version = str(instance["version"])
    return f"{repo}__{version}"


def get_instance_docker_image(instance_id: str) -> str:
    """Get the Docker image for a SWE-fficiency instance."""
    return f"{constants.DOCKER_IMAGE_PREFIX}:{instance_id}"


def get_instruction(
    instance: dict,
    metadata: EvalMetadata,
    workspace_path: str,
) -> str:
    """Generate instruction for the agent."""
    workspace_dir_name = get_swefficiency_workspace_dir_name(instance)

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

    # Render the instruction
    instruction = template.render(context)
    return instruction


def divide_cpus_among_workers(
    num_workers: int,
    num_cpus_per_worker: int = 4,
    num_to_skip: int = 0,
) -> list[list[int]]:
    """Divide CPUs among workers for resource isolation.

    Args:
        num_workers: Number of parallel workers.
        num_cpus_per_worker: CPUs to allocate per worker.
        num_to_skip: Number of CPUs to skip at the beginning.

    Returns:
        List of CPU lists, one per worker.
    """
    try:
        current_cpus = list(os.sched_getaffinity(0))
    except AttributeError:
        # os.sched_getaffinity not available on all platforms
        current_cpus = list(range(multiprocessing.cpu_count()))

    num_cpus = len(current_cpus)
    if num_workers <= 0:
        raise ValueError("Number of workers must be greater than 0")

    total_cpus_needed = num_workers * num_cpus_per_worker + num_to_skip
    if total_cpus_needed > num_cpus:
        raise ValueError(
            f"Not enough CPUs available. Requested {total_cpus_needed} "
            f"CPUs (num_workers={num_workers}, num_cpus_per_worker={num_cpus_per_worker}, "
            f"num_to_skip={num_to_skip}), but only {num_cpus} CPUs are available."
        )

    available_cpus = current_cpus[num_to_skip:]
    cpu_groups = [
        available_cpus[i * num_cpus_per_worker : (i + 1) * num_cpus_per_worker]
        for i in range(num_workers)
    ]
    logger.info(
        f"Divided {num_cpus} CPUs into {num_workers} groups, "
        f"each with {num_cpus_per_worker} CPUs: {cpu_groups}"
    )
    return cpu_groups


class SWEfficiencyEvaluation(Evaluation):
    """
    Process-based SWE-fficiency evaluation implemented as a child of the
    abstract Evaluation orchestrator.

    Implements:
      - prepare_instances()
      - prepare_workspace(instance)
      - evaluate_instance(instance, workspace)
    """

    # CPU group management for Docker resource limits
    cpu_groups_queue: Any = Field(
        default=None,
        description="Queue of CPU groups for worker resource allocation",
    )
    num_cpus_per_worker: int = Field(
        default=DOCKER_DEFAULTS["num_cpus_per_worker"],
        description="Number of CPUs per worker",
    )
    mem_limit: str = Field(
        default=DOCKER_DEFAULTS["mem_limit"],
        description="Memory limit per container",
    )

    model_config = {"arbitrary_types_allowed": True}

    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up SWE-fficiency evaluation data")

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

    def _acquire_cpu_group(self) -> list[int] | None:
        """Acquire a CPU group from the queue for this worker."""
        if self.cpu_groups_queue is not None:
            try:
                cpu_group = self.cpu_groups_queue.get_nowait()
                logger.info(f"Worker acquired CPU group: {cpu_group}")
                return cpu_group
            except Exception:
                logger.warning("Failed to get CPU group from queue")
        return None

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """
        Create workspace for the instance.

        For Docker workspace, builds agent-server image from base swefficiency image.
        For remote workspace, uses APIRemoteWorkspace with pre-built images.
        """
        # Get base swefficiency image
        base_docker_image = get_instance_docker_image(instance.id)
        build_target = constants.DEFAULT_BUILD_TARGET
        custom_tag = f"swefficiency.{instance.id}"

        # Build agent server image tag
        suffix = f"-{build_target}" if build_target != "binary" else ""
        agent_server_image = (
            f"{EVAL_AGENT_SERVER_IMAGE}:{SDK_SHORT_SHA}-{custom_tag}{suffix}"
        )

        logger.info(f"Base image: {base_docker_image}")
        logger.info(f"Agent server image: {agent_server_image}")

        if self.metadata.workspace_type == "docker":
            # Build agent-server image from base swefficiency image
            SKIP_BUILD = os.getenv("SKIP_BUILD", "0").lower() in ("1", "true", "yes")
            logger.info(f"SKIP_BUILD={SKIP_BUILD}")

            if not SKIP_BUILD:
                logger.info(
                    f"Building workspace from {base_docker_image} "
                    f"for instance {instance.id}. "
                    "This may take a while..."
                )
                output = build_image(
                    base_image=base_docker_image,
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

            # Get CPU group for resource limiting
            cpu_group = self._acquire_cpu_group()

            # Build workspace kwargs with resource limits
            workspace_kwargs: dict[str, Any] = {
                "server_image": agent_server_image,
                "working_dir": "/workspace",
                "forward_env": forward_env or [],
                "mem_limit": self.mem_limit,
            }

            if cpu_group is not None:
                workspace_kwargs["cpuset_cpus"] = ",".join(map(str, cpu_group))
                workspace_kwargs["nano_cpus"] = int(1e9 * len(cpu_group))

            workspace = ResourceLimitedDockerWorkspace(**workspace_kwargs)

            # Store CPU group and queue on workspace for automatic cleanup
            workspace._cpu_group = cpu_group
            workspace._cpu_groups_queue = self.cpu_groups_queue

        elif self.metadata.workspace_type == "remote":
            runtime_api_key = os.getenv("RUNTIME_API_KEY")
            sdk_short_sha = os.getenv("SDK_SHORT_SHA", SDK_SHORT_SHA)
            if not runtime_api_key:
                raise ValueError(
                    "RUNTIME_API_KEY environment variable is not set for remote workspace"
                )

            # For remote, use SDK_SHORT_SHA from env if available
            remote_agent_image = (
                f"{EVAL_AGENT_SERVER_IMAGE}:{sdk_short_sha}-{custom_tag}{suffix}"
            )
            if not image_exists(remote_agent_image):
                raise RuntimeError(
                    f"Agent server image {remote_agent_image} does not exist in container registry, "
                    "make sure to build, push it, and make it public accessible before using remote workspace."
                )

            logger.info(
                f"Using remote workspace with image {remote_agent_image} "
                f"(sdk sha: {sdk_short_sha}, resource_factor: {resource_factor})"
            )

            workspace = APIRemoteWorkspace(
                runtime_api_url=os.getenv(
                    "RUNTIME_API_URL", "https://runtime.eval.all-hands.dev"
                ),
                runtime_api_key=runtime_api_key,
                server_image=remote_agent_image,
                target_type="source",
                forward_env=forward_env or [],
                resource_factor=resource_factor,
                init_timeout=600.0,
                startup_wait_timeout=600.0,
            )
        else:
            raise ValueError(
                f"Unsupported workspace_type: {self.metadata.workspace_type}"
            )

        # Run env setup commands
        for cmd in self.metadata.env_setup_commands or []:
            res = workspace.execute_command(cmd)
            if res.exit_code != 0:
                raise RuntimeError(
                    f"Failed to run env setup command '{cmd}': {res.stderr}"
                )
            logger.debug(f"Ran env setup command '{cmd}': {res.stdout}")

        return workspace

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """
        Create conversation, run agent, collect history and git patch.
        """
        tools = get_default_tools(enable_browser=False)
        agent = Agent(
            llm=self.metadata.llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
        )

        assert isinstance(workspace, RemoteWorkspace)

        # Set up workspace directory
        workspace_dir_name = get_swefficiency_workspace_dir_name(instance.data)
        repo_path = f"/workspace/{workspace_dir_name}/"
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

        # Copy testbed to workspace
        logger.info("repo_path: %s", repo_path)
        cp_testbed_repo = workspace.execute_command(
            f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}"
        )
        assert cp_testbed_repo.exit_code == 0, (
            f"cp_testbed_repo failed: {cp_testbed_repo.stderr}"
        )

        # git reset
        git_reset = workspace.execute_command(f"cd {repo_path} ; git reset --hard")
        assert git_reset.exit_code == 0, f"git reset failed: {git_reset.stderr}"

        # Remove git remotes
        workspace.execute_command(
            f"cd {repo_path} ; "
            'for remote_name in $(git remote); do git remote remove "$remote_name"; done'
        )

        # Get instruction and run conversation
        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path=workspace.working_dir,
        )
        conversation.send_message(instruction)
        run_conversation_with_fake_user_response(conversation)

        # git add
        workspace.execute_command(f"cd {repo_path} ; git add -A")

        # Remove binary files from staging
        workspace.execute_command(
            f"cd {repo_path} ; "
            'for file in $(git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" | cut -c4-); do '
            'if [ -f "$file" ] && (file "$file" | grep -q "executable" || '
            'git check-attr binary "$file" | grep -q "binary: set"); then '
            'git rm -f "$file" 2>/dev/null || rm -f "$file"; '
            "fi; done"
        )

        # git commit
        workspace.execute_command(
            f"cd {repo_path} && "
            f"git config --global user.email '{constants.GIT_USER_EMAIL}' && "
            f"git config --global user.name '{constants.GIT_USER_NAME}' && "
            f"git commit --no-verify -m '{constants.GIT_COMMIT_MESSAGE}'"
        )

        # Get git patch
        base_commit = instance.data["base_commit"]
        git_patch_result = workspace.execute_command(
            f"cd {repo_path} ; git --no-pager diff --no-color {base_commit} HEAD"
        )
        assert git_patch_result.exit_code == 0, (
            f"git diff failed: {git_patch_result.stderr}"
        )
        git_patch = git_patch_result.stdout

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
    parser.add_argument(
        "--num-cpus-per-worker",
        type=int,
        default=DOCKER_DEFAULTS["num_cpus_per_worker"],
        help="Number of CPUs per worker for Docker resource limits",
    )
    parser.add_argument(
        "--mem-limit",
        type=str,
        default=DOCKER_DEFAULTS["mem_limit"],
        help="Memory limit per Docker container (e.g., '16g')",
    )
    parser.add_argument(
        "--num-cpus-to-skip",
        type=int,
        default=DOCKER_DEFAULTS["num_cpus_to_skip"],
        help="Number of CPUs to skip at the beginning",
    )
    parser.set_defaults(**INFER_DEFAULTS)
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
        details={},
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
        max_attempts=args.max_attempts,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
    )

    # Set up CPU groups queue for Docker workspace
    cpu_groups_queue = None
    if args.workspace == "docker" and args.num_workers > 1:
        try:
            cpu_groups = divide_cpus_among_workers(
                num_workers=args.num_workers,
                num_cpus_per_worker=args.num_cpus_per_worker,
                num_to_skip=args.num_cpus_to_skip,
            )
            cpu_groups_queue = multiprocessing.Manager().Queue()
            for cpu_group in cpu_groups:
                cpu_groups_queue.put(cpu_group)
            logger.info(f"Initialized CPU groups queue with {len(cpu_groups)} groups")
        except ValueError as e:
            logger.warning(
                f"Could not set up CPU groups: {e}. Running without CPU pinning."
            )

    # Run orchestrator
    evaluator = SWEfficiencyEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
        cpu_groups_queue=cpu_groups_queue,
        num_cpus_per_worker=args.num_cpus_per_worker,
        mem_limit=args.mem_limit,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")
    print(json.dumps({"output_json": str(evaluator.output_path)}))


if __name__ == "__main__":
    main()
