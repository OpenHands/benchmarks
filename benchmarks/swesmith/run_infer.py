import json
import os
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader

from benchmarks.swesmith import constants
from benchmarks.swesmith.build_images import (
    extract_custom_tag,
    get_official_docker_image,
)
from benchmarks.swesmith.config import INFER_DEFAULTS
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
from openhands.workspace import APIRemoteWorkspace, DockerWorkspace


logger = get_logger(__name__)

_SSH_KEY_CONTAINER_PATH = "/workspace/github_key"
_GIT_SSH_COMMAND = (
    f"ssh -i {_SSH_KEY_CONTAINER_PATH}"
    " -o StrictHostKeyChecking=accept-new"
    " -o IdentitiesOnly=yes"
)

_DEFAULT_SSH_KEYS = [
    "id_rsa",
    "id_ecdsa",
    "id_ecdsa_sk",
    "id_ed25519",
    "id_ed25519_sk",
    "id_xmss",
]


def _find_ssh_key() -> Path | None:
    """Find an SSH private key: GITHUB_USER_SSH_KEY env var first, then default paths."""
    key_path = os.environ.get("GITHUB_USER_SSH_KEY")
    if key_path and Path(key_path).exists():
        return Path(key_path)

    ssh_dir = Path.home() / ".ssh"
    for key_name in _DEFAULT_SSH_KEYS:
        key_file = ssh_dir / key_name
        if key_file.exists():
            return key_file

    return None


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


class SWESmithEvaluation(Evaluation):
    """
    Process-based SWE-Smith evaluation implemented as a child of the
    abstract Evaluation orchestrator.

    Implements:
      - prepare_instances()
      - prepare_workspace(instance)
      - evaluate_instance(instance, workspace)
    """

    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up SWE-Smith evaluation data")

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
        """
        # ADAPTATION 1: Use image_name field from dataset instead of deriving
        # from instance_id (SWE-Smith stores image name directly in dataset)
        official_docker_image = get_official_docker_image(instance.data["image_name"])
        build_target = constants.DEFAULT_BUILD_TARGET
        custom_tag = extract_custom_tag(official_docker_image)
        # For non-binary targets, append target suffix
        suffix = (
            f"-{build_target}" if build_target != constants.BUILD_TARGET_BINARY else ""
        )
        base_agent_image = (
            f"{EVAL_AGENT_SERVER_IMAGE}:{SDK_SHORT_SHA}-{custom_tag}{suffix}"
        )
        agent_server_image = base_agent_image

        # Forward all OPENHANDS_* env vars into the container with prefix stripped.
        # e.g. OPENHANDS_ANTHROPIC_API_KEY becomes ANTHROPIC_API_KEY inside the container.
        OPENHANDS_ENV_PREFIX = "OPENHANDS_"
        forwarded_env_names = []
        for key, value in os.environ.items():
            if key.startswith(OPENHANDS_ENV_PREFIX):
                stripped = key[len(OPENHANDS_ENV_PREFIX) :]
                os.environ[stripped] = value
                forwarded_env_names.append(stripped)
        all_forward_env = list(forward_env or []) + forwarded_env_names

        volumes = []

        # Forward GIT_SSH_COMMAND for private repo git fetch.
        # The actual key is injected in evaluate_instance() via base64 to avoid
        # Docker bind-mount permission issues.
        ssh_key_path = _find_ssh_key()
        if ssh_key_path:
            all_forward_env.append("GIT_SSH_COMMAND")
            os.environ["GIT_SSH_COMMAND"] = _GIT_SSH_COMMAND
            logger.info(f"Found SSH key {ssh_key_path} for private repo access")

        if self.metadata.workspace_type == "docker":
            SKIP_BUILD = os.getenv("SKIP_BUILD", "1").lower() in ("1", "true", "yes")
            logger.info(f"SKIP_BUILD={SKIP_BUILD}")
            if not SKIP_BUILD:
                logger.info(
                    f"Building workspace from {official_docker_image} "
                    f"for instance {instance.id}. "
                    "This may take a while...\n"
                    "You can run benchmarks/swesmith/build_images.py and set "
                    "SKIP_BUILD=1 to skip building and use pre-built "
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
                if base_agent_image not in output.tags:
                    raise RuntimeError(
                        f"Built image tags {output.tags} do not include expected tag "
                        f"{base_agent_image}"
                    )

            workspace = DockerWorkspace(
                server_image=agent_server_image,
                working_dir="/workspace",
                forward_env=all_forward_env,
                volumes=volumes,
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
                f"Using remote workspace with image {agent_server_image} "
                f"(sdk sha: {sdk_short_sha}, resource_factor: {resource_factor})"
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
        tools = get_default_tools(
            # Disable browser tools in CLI mode
            enable_browser=False,
        )
        agent = Agent(
            llm=self.metadata.llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
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
        cp_testebed_repo = workspace.execute_command(
            (f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}")
        )
        assert cp_testebed_repo.exit_code == 0, (
            f"cp_testebed_repo failed: {cp_testebed_repo.stderr}"
        )

        # git reset
        git_reset = workspace.execute_command(f"cd {repo_path} ; git reset --hard")
        assert git_reset.exit_code == 0, f"git reset failed: {git_reset.stderr}"

        # Inject SSH key into container for private repo git fetch.
        # We base64-encode and decode to avoid shell escaping issues and
        # Docker bind-mount permission problems.
        ssh_key = _find_ssh_key()
        if ssh_key:
            import base64

            key_b64 = base64.b64encode(ssh_key.read_bytes()).decode()
            setup_ssh = workspace.execute_command(
                f"echo '{key_b64}' | base64 -d > {_SSH_KEY_CONTAINER_PATH}"
                f" && chmod 600 {_SSH_KEY_CONTAINER_PATH}"
            )
            assert setup_ssh.exit_code == 0, f"SSH key setup failed: {setup_ssh.stderr}"

        # Fetch bug branch from GitHub mirror and checkout
        # Use SSH URL for private repos when an SSH key is available
        if ssh_key:
            mirror_url = f"git@github.com:{instance.data['repo']}.git"
        else:
            mirror_url = f"https://github.com/{instance.data['repo']}.git"
        git_fetch = workspace.execute_command(
            f"cd {repo_path} ; git fetch {mirror_url} {instance.id}"
        )
        assert git_fetch.exit_code == 0, f"git fetch failed: {git_fetch.stderr}"
        git_checkout = workspace.execute_command(
            f"cd {repo_path} ; git checkout FETCH_HEAD"
        )
        assert git_checkout.exit_code == 0, (
            f"git checkout failed: {git_checkout.stderr}"
        )

        # Remove untracked files (respects .gitignore, so installed deps are preserved)
        workspace.execute_command(f"cd {repo_path} ; git clean -fdq")

        # Capture HEAD after checkout so base_commit reflects the bug branch
        head_result = workspace.execute_command(f"cd {repo_path} ; git rev-parse HEAD")
        assert head_result.exit_code == 0, (
            f"git rev-parse HEAD failed: {head_result.stderr}"
        )
        base_commit = head_result.stdout.strip()
        instance.data["base_commit"] = base_commit

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path=workspace.working_dir,
        )
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
        git_patch_result = workspace.execute_command(
            (f"cd {repo_path} ; git --no-pager diff --no-color {base_commit} HEAD")
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
    from dotenv import load_dotenv

    load_dotenv()

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

    # Run orchestrator with a simple JSONL writer
    evaluator = SWESmithEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")
    # Emit machine-readable path for callers
    print(json.dumps({"output_json": str(evaluator.output_path)}))


if __name__ == "__main__":
    main()
