import os
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader

from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
)
from openhands.agent_server.docker.build import SDK_VERSION, _base_slug
from openhands.sdk import LLM, Agent, Conversation, get_logger
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerWorkspace


logger = get_logger(__name__)


def get_official_docker_image(
    instance_id: str,
    docker_image_prefix="docker.io/swebench/",
) -> str:
    # Official SWE-Bench image
    # swebench/sweb.eval.x86_64.django_1776_django-11333:v1
    repo, name = instance_id.split("__")
    official_image_name = docker_image_prefix.rstrip("/")
    official_image_name += f"/sweb.eval.x86_64.{repo}_1776_{name}:latest".lower()
    logger.debug(f"Official SWE-Bench image: {official_image_name}")
    return official_image_name


def get_agent_server_docker_image(
    instance_id: str,
    docker_image_prefix="docker.io/swebench/",
    target: str = "source-minimal",
) -> str:
    official_image_name = get_official_docker_image(instance_id, docker_image_prefix)
    return (
        "ghcr.io/openhands/agent-server"
        + f":v{SDK_VERSION}_{_base_slug(official_image_name)}_{target}"
    )


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
        SKIP_BUILD = os.getenv("SKIP_BUILD", "1").lower() in ("1", "true", "yes")
        logger.info(f"SKIP_BUILD={SKIP_BUILD}")
        if SKIP_BUILD:
            agent_server_image = get_agent_server_docker_image(instance.id)
            workspace = DockerWorkspace(
                server_image=agent_server_image,
                working_dir="/workspace",
            )
        else:
            official_docker_image = get_official_docker_image(instance.id)
            workspace = DockerWorkspace(
                base_image=official_docker_image,
                working_dir="/workspace",
                target="source-minimal",
            )
            logger.info(
                f"Building workspace from {official_docker_image}. "
                "This may take a while...\n"
                "You can run benchmarks/swe_bench/build_images.py and set "
                "SWE_BENCH_SKIP_BUILD=1 to skip building and use pre-built "
                "agent-server image."
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
        conversation.send_message(instruction)
        conversation.run()

        # Collect results
        history = list(map(lambda event: event.model_dump(), conversation.state.events))

        # git add
        workspace.execute_command(f"cd {repo_path} ; git add -A")

        # git commit
        workspace.execute_command(
            f"cd {repo_path} && "
            "git config --global user.email 'evaluation@openhands.dev' && "
            "git config --global user.name 'OpenHands Evaluation' && "
            "git commit -m 'patch'"
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

        # EvalOutput is your model; keep fields consistent with prior JSONL
        out = EvalOutput(
            instance_id=instance.id,
            test_result={
                "git_patch": git_patch,
            },
            instruction=instruction,
            error=None,
            history=history,
            metrics=conversation.conversation_stats.get_combined_metrics().model_dump(),
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
        critic_name=args.critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
    )

    # Run orchestrator with a simple JSONL writer
    evaluator = SWEBenchEvaluation(metadata=metadata, num_workers=args.num_workers)

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
