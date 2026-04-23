"""Run inference for the Hybrid-Gym func_localize benchmark.

Task: Agent must locate a function/class by description (no file path given)
and add a docstring to it. Uses a generic python:3.11-bookworm image,
cloning the repository at runtime.

Dataset: hybrid-gym/hybrid_gym_func_localize on HuggingFace.
"""

import json
import os
from typing import Any, List

from jinja2 import Environment, FileSystemLoader

from benchmarks.hybridgym_funclocalize.config import INFER_DEFAULTS
from benchmarks.utils.args_parser import add_prompt_path_argument, get_parser
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
from benchmarks.utils.image_utils import create_docker_workspace, remote_image_exists
from benchmarks.utils.litellm_proxy import build_eval_llm
from benchmarks.utils.llm_config import load_llm_config
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
    ToolPresetType,
)
from benchmarks.utils.version import IMAGE_TAG_PREFIX
from openhands.sdk import Agent, Conversation, Tool, get_logger
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.delegate import DelegateTool
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import APIRemoteWorkspace


logger = get_logger(__name__)

# Docker image for the workspace — generic Python, no specialized environment needed.
BASE_DOCKER_IMAGE = "python:3.11-bookworm"


def _get_workspace_dir_name(instance: dict) -> str:
    """Derive workspace directory name from repo and base_commit."""
    repo = instance["repo"]
    commit = instance.get("base_commit", "latest")[:8]
    return f"{repo}__{commit}".replace("/", "__")


def _get_docstring_remove_command(instance: dict) -> str:
    """Generate sed command(s) to remove existing docstrings before the agent runs."""
    if "functions" in instance and instance["functions"]:
        commands = []
        for func in instance["functions"]:
            start = func["docstring_line_start"]
            end = func["docstring_line_end"]
            if start != -1:
                commands.append(f"sed -i '{start + 1},{end + 1}d' {func['file_path']}")
        return " && ".join(commands) if commands else ""

    start = instance.get("docstring_line_start", -1)
    end = instance.get("docstring_line_end", -1)
    file_path = instance.get("file_path", "")
    if start != -1 and file_path:
        return f"sed -i '{start + 1},{end + 1}d' {file_path}"
    return ""


def _format_targets(functions: list[dict]) -> str:
    """Format a list of target functions for a multi-function prompt."""
    lines = []
    for idx, func in enumerate(functions, start=1):
        name = func.get("module_name", "<unknown>")
        kind = func.get("module_type", "function")
        entry = f"{idx}. {name} ({kind})"
        desc = func.get("function_description")
        if desc:
            entry += f"\n   Description: {desc}"
        lines.append(entry)
    return "\n".join(lines)


def get_instruction(instance: dict, metadata: EvalMetadata, workspace_path: str) -> str:
    """Generate the task instruction for the agent."""
    workspace_dir_name = instance.get(
        "_workspace_dir_name", _get_workspace_dir_name(instance)
    )

    assert metadata.prompt_path is not None
    prompts_dir = os.path.dirname(metadata.prompt_path)
    template_name = os.path.basename(metadata.prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Multi-function instances carry a 'functions' list; single-function
    # instances describe one target via top-level fields.
    functions = instance.get("functions")
    is_multi = bool(functions)

    description = (
        instance.get("brief_description")
        or instance.get("function_description")
        or instance.get("description")
        or "No description provided."
    )

    context = {
        "instance": instance,
        "workspace_dir_name": workspace_dir_name,
        "actual_workspace_path": workspace_path,
        "module_type": instance.get("module_type", "function"),
        "description": description,
        "target_name": instance.get("module_name", "<unknown>"),
        "metadata": metadata,
        # Multi-function fields
        "is_multi": is_multi,
        "num_targets": len(functions) if is_multi else 1,
        "target_list": _format_targets(functions) if is_multi else "",
    }

    return template.render(context)


class FuncLocalizeEvaluation(Evaluation):
    """
    Hybrid-Gym func_localize evaluation implemented as a child of the
    abstract Evaluation orchestrator.

    Implements:
      - prepare_instances()
      - prepare_workspace(instance)
      - evaluate_instance(instance, workspace)
    """

    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up func_localize evaluation data")

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

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """Create a generic Docker workspace; clone repo at runtime."""
        agent_server_image = f"{EVAL_AGENT_SERVER_IMAGE}:{IMAGE_TAG_PREFIX}-hybridgym-funclocalize-binary"

        if self.metadata.workspace_type == "docker":
            workspace = create_docker_workspace(
                agent_server_image=agent_server_image,
                base_image=BASE_DOCKER_IMAGE,
                build_target="binary",
                forward_env=forward_env or [],
            )
        elif self.metadata.workspace_type == "remote":
            runtime_api_key = os.getenv("RUNTIME_API_KEY")
            if not runtime_api_key:
                raise ValueError(
                    "RUNTIME_API_KEY environment variable is not set for remote workspace"
                )
            if not remote_image_exists(agent_server_image):
                raise RuntimeError(
                    f"Agent server image {agent_server_image} does not exist in container registry."
                )
            startup_timeout = float(os.getenv("REMOTE_RUNTIME_STARTUP_TIMEOUT", "600"))
            workspace = APIRemoteWorkspace(
                runtime_api_url=os.getenv(
                    "RUNTIME_API_URL", "https://runtime.eval.all-hands.dev"
                ),
                runtime_api_key=runtime_api_key,
                server_image=agent_server_image,
                target_type="binary",
                resource_factor=resource_factor,
                forward_env=forward_env or [],
                init_timeout=startup_timeout,
                startup_wait_timeout=startup_timeout,
            )
        else:
            raise ValueError(
                f"Unsupported workspace_type: {self.metadata.workspace_type}"
            )

        return workspace

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """Clone repo, remove docstrings, run agent, collect patch."""
        agent_llm = build_eval_llm(self.metadata.llm)
        tools = self._get_tools(preset=self.metadata.tool_preset)
        if self.metadata.enable_delegation:
            tools.append(Tool(name=DelegateTool.name))
        condenser = None
        if self.metadata.enable_condenser:
            condenser = LLMSummarizingCondenser(
                llm=build_eval_llm(self.metadata.llm, usage_id="condenser"),
                max_size=self.metadata.condenser_max_size,
                keep_first=self.metadata.condenser_keep_first,
            )
        agent = Agent(
            llm=agent_llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
            condenser=condenser,
        )

        assert isinstance(workspace, RemoteWorkspace)

        # --- Workspace initialization ---
        workspace_dir_name = _get_workspace_dir_name(instance.data)
        instance.data["_workspace_dir_name"] = workspace_dir_name
        repo_path = f"/workspace/{workspace_dir_name}"

        # Clone repository
        res = workspace.execute_command(
            f"git clone https://github.com/{instance.data['repo']}.git {repo_path}"
        )
        if res.exit_code != 0:
            raise RuntimeError(f"Failed to clone repo: {res.stderr}")

        # Checkout base commit
        res = workspace.execute_command(
            f"cd {repo_path} && git checkout {instance.data['base_commit']} && git reset --hard"
        )
        if res.exit_code != 0:
            raise RuntimeError(f"Failed to checkout base commit: {res.stderr}")

        # Remove git remotes
        workspace.execute_command(
            f"cd {repo_path} && for r in $(git remote); do git remote remove $r; done"
        )

        # Remove existing docstrings
        remove_cmd = _get_docstring_remove_command(instance.data)
        if remove_cmd:
            res = workspace.execute_command(f"cd {repo_path} && {remove_cmd}")
            if res.exit_code != 0:
                raise RuntimeError(f"Failed to remove docstrings: {res.stderr}")

        # Re-init git to track agent changes from a clean slate
        workspace.execute_command(f"cd {repo_path} && rm -rf .git")
        res = workspace.execute_command(
            f"cd {repo_path} && git init . && git add -A && "
            f"git config user.email 'eval@openhands.dev' && "
            f"git config user.name 'OpenHands Eval' && "
            f"git commit -m 'Initial commit'"
        )
        if res.exit_code != 0:
            logger.warning("Git re-init failed: %s", res.stderr)

        # Capture new HEAD as base_commit for later diff
        head_res = workspace.execute_command(f"cd {repo_path} && git rev-parse HEAD")
        if head_res.exit_code == 0:
            instance.data["base_commit"] = head_res.stdout.strip()
            logger.info("Captured base_commit: %s", instance.data["base_commit"])

        # --- Run agent ---
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

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path=workspace.working_dir,
        )
        conversation.send_message(instruction)
        run_conversation_with_fake_user_response(conversation)

        # --- Extract git patch ---
        workspace.execute_command(f"cd {repo_path} && git add -A")

        # Commit agent changes (may fail if no changes, that's ok)
        workspace.execute_command(
            f"cd {repo_path} && git commit --no-verify -m 'Agent changes' || true"
        )

        base_commit = instance.data["base_commit"]
        logger.info("Extracting git patch: base_commit=%s", base_commit)

        git_patch_result = workspace.execute_command(
            f"cd {repo_path} && git --no-pager diff --no-color {base_commit} HEAD"
        )
        if git_patch_result.exit_code != 0:
            logger.warning(
                "git diff failed (exit=%d): %s",
                git_patch_result.exit_code,
                git_patch_result.stderr,
            )
        git_patch = git_patch_result.stdout if git_patch_result.exit_code == 0 else ""
        logger.info("Git patch length: %d chars", len(git_patch))

        summarize_instance(
            instance_id=instance.id,
            conversation=conversation,
            git_patch=git_patch,
            logger=logger,
        )

        test_result: dict[str, Any] = {
            "git_patch": git_patch,
        }

        return EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result=test_result,
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )

    def _get_tools(self, preset: ToolPresetType = "default") -> list[Tool]:
        """Get tools for the given preset."""
        if preset == "gemini":
            from openhands.tools.preset.gemini import get_gemini_tools

            return get_gemini_tools(enable_browser=False)
        elif preset == "gpt5":
            from openhands.tools.preset.gpt5 import get_gpt5_tools

            return get_gpt5_tools(enable_browser=False)
        elif preset == "planning":
            from openhands.tools.preset.planning import get_planning_tools

            return get_planning_tools()
        else:
            return get_default_tools(enable_browser=False)


def main() -> None:
    parser = get_parser()
    add_prompt_path_argument(parser, __file__)
    parser.set_defaults(**INFER_DEFAULTS)
    args = parser.parse_args()

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

    critic = create_critic(args)

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

    evaluator = FuncLocalizeEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")
    print(json.dumps({"output_json": str(evaluator.output_path)}))


if __name__ == "__main__":
    main()
