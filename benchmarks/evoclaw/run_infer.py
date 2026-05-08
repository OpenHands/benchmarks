from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader

from benchmarks.evoclaw.config import INFER_DEFAULTS
from benchmarks.utils.agent_context import create_agent_context
from benchmarks.utils.args_parser import add_prompt_path_argument, get_parser
from benchmarks.utils.console_logging import summarize_instance
from benchmarks.utils.conversation import build_event_persistence_callback
from benchmarks.utils.critics import create_critic
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response
from benchmarks.utils.litellm_proxy import build_eval_llm
from benchmarks.utils.llm_config import load_llm_config
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
    ToolPresetType,
)
from openhands.sdk import Agent, Conversation, Tool, get_logger
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.delegate import DelegateTool
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerDevWorkspace


logger = get_logger(__name__)


def get_tools_for_preset(
    preset: ToolPresetType,
    enable_browser: bool = False,
) -> list[Tool]:
    if preset == "gemini":
        from openhands.tools.preset.gemini import get_gemini_tools

        return get_gemini_tools(enable_browser=enable_browser)
    if preset == "gpt5":
        from openhands.tools.preset.gpt5 import get_gpt5_tools

        return get_gpt5_tools(enable_browser=enable_browser)
    if preset == "planning":
        from openhands.tools.preset.planning import get_planning_tools

        return get_planning_tools()

    return get_default_tools(enable_browser=enable_browser)


def _repo_image_name(repo_name: str) -> str:
    return f"{repo_name.lower()}/base:latest"


def _read_selected_milestones(repo_root: Path) -> list[str]:
    selected = repo_root / "selected_milestone_ids.txt"
    if selected.exists():
        return [
            line.strip() for line in selected.read_text().splitlines() if line.strip()
        ]

    milestones = repo_root / "milestones.csv"
    if milestones.exists():
        with milestones.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            key = "milestone_id" if "milestone_id" in fieldnames else fieldnames[0]
            return [row[key].strip() for row in reader if row.get(key, "").strip()]

    dependencies = repo_root / "dependencies.csv"
    if dependencies.exists():
        ids: set[str] = set()
        with dependencies.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for value in row.values():
                    value = (value or "").strip()
                    if value:
                        ids.add(value)
        return sorted(ids)

    return []


def _write_workspace_file(
    workspace: RemoteWorkspace,
    local_path: Path,
    remote_path: str,
) -> None:
    result = workspace.file_upload(local_path, remote_path)
    if not result.success:
        raise RuntimeError(f"Failed to upload {local_path} to {remote_path}: {result}")


def _render_instruction(
    prompt_path: str,
    task_queue_path: str,
    srs_dir: str,
) -> str:
    prompts_dir = os.path.dirname(prompt_path)
    template_name = os.path.basename(prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    return env.get_template(template_name).render(
        task_queue_path=task_queue_path,
        srs_dir=srs_dir,
    )


class EvoClawEvaluation(Evaluation):
    def prepare_instances(self) -> List[EvalInstance]:
        assert self.metadata.details is not None
        data_root = Path(self.metadata.details["data_root"]).expanduser().resolve()
        selected_repos = self.metadata.details.get("selected_repos")

        instances: list[EvalInstance] = []
        for repo_root in sorted(data_root.iterdir()):
            if not repo_root.is_dir() or not (repo_root / "metadata.json").exists():
                continue
            if selected_repos and not any(
                selected in repo_root.name for selected in selected_repos
            ):
                continue

            milestone_ids = _read_selected_milestones(repo_root)
            instances.append(
                EvalInstance(
                    id=repo_root.name,
                    data={
                        "repo_root": str(repo_root),
                        "image": _repo_image_name(repo_root.name),
                        "milestone_ids": milestone_ids,
                    },
                )
            )

        if self.metadata.eval_limit:
            instances = instances[: self.metadata.eval_limit]
        logger.info("Prepared %d EvoClaw instances", len(instances))
        return instances

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        return DockerDevWorkspace(
            base_image=instance.data["image"],
            target="source",
            working_dir="/testbed",
            forward_env=forward_env or [],
        )

    def _upload_task_materials(
        self,
        instance: EvalInstance,
        workspace: RemoteWorkspace,
    ) -> dict[str, str]:
        repo_root = Path(instance.data["repo_root"])
        material_dir = Path(self.metadata.eval_output_dir) / "evoclaw_materials"
        material_dir.mkdir(parents=True, exist_ok=True)
        instance_dir = material_dir / instance.id
        instance_dir.mkdir(parents=True, exist_ok=True)

        remote_srs_dir = "/e2e_workspace/srs"
        remote_task_queue = "/e2e_workspace/TASK_QUEUE.md"
        workspace.execute_command("mkdir -p /e2e_workspace/srs", timeout=30)

        queue_lines = [
            "# EvoClaw Task Queue",
            "",
            "Implement the following milestones in /testbed:",
            "",
        ]
        for milestone_id in instance.data["milestone_ids"]:
            srs_path = repo_root / "srs" / milestone_id / "SRS.md"
            if not srs_path.exists():
                logger.warning("Missing SRS for %s: %s", milestone_id, srs_path)
                continue
            local_srs = instance_dir / f"{milestone_id}_SRS.md"
            local_srs.write_text(srs_path.read_text(encoding="utf-8"), encoding="utf-8")
            remote_srs = f"{remote_srs_dir}/{milestone_id}_SRS.md"
            _write_workspace_file(workspace, local_srs, remote_srs)
            queue_lines.append(f"- {milestone_id}: {remote_srs}")

        local_queue = instance_dir / "TASK_QUEUE.md"
        local_queue.write_text("\n".join(queue_lines) + "\n", encoding="utf-8")
        _write_workspace_file(workspace, local_queue, remote_task_queue)

        return {
            "task_queue_path": remote_task_queue,
            "srs_dir": remote_srs_dir,
        }

    def evaluate_instance(
        self,
        instance: EvalInstance,
        workspace: RemoteWorkspace,
    ) -> EvalOutput:
        agent_llm = build_eval_llm(self.metadata.llm)
        tools = get_tools_for_preset(
            preset=self.metadata.tool_preset,
            enable_browser=False,
        )
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
            agent_context=create_agent_context(),
        )

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

        paths = self._upload_task_materials(instance, workspace)
        assert self.metadata.prompt_path is not None
        instruction = _render_instruction(
            prompt_path=self.metadata.prompt_path,
            task_queue_path=paths["task_queue_path"],
            srs_dir=paths["srs_dir"],
        )

        workspace.execute_command("cd /testbed && git reset --hard", timeout=120)
        conversation.send_message(instruction)
        run_conversation_with_fake_user_response(conversation)

        diff_result = workspace.execute_command(
            "cd /testbed && git --no-pager diff --no-color",
            timeout=120,
        )
        if diff_result.exit_code != 0:
            raise RuntimeError(f"git diff failed: {diff_result.stderr}")
        git_patch = diff_result.stdout

        summarize_instance(
            instance_id=instance.id,
            conversation=conversation,
            git_patch=git_patch,
            logger=logger,
        )

        return EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result={"git_patch": git_patch},
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=conversation.conversation_stats.get_combined_metrics(),
            instance=instance.data,
        )


def main() -> None:
    parser = get_parser()
    add_prompt_path_argument(parser, __file__)
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to EvoClaw-data containing repo directories with metadata.json.",
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        default=None,
        help="Optional repo-name substring filters, e.g. --repos navidrome ripgrep.",
    )
    parser.set_defaults(**INFER_DEFAULTS)
    args = parser.parse_args()

    llm = load_llm_config(args.llm_config_path)
    selected_repos = args.repos
    if args.select:
        selected_from_file = [
            line.strip()
            for line in Path(args.select).read_text().splitlines()
            if line.strip()
        ]
        selected_repos = (selected_repos or []) + selected_from_file

    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name="evoclaw",
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )

    enable_condenser = args.enable_condenser
    if args.disable_condenser:
        enable_condenser = False

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={
            "data_root": str(Path(args.data_root).expanduser().resolve()),
            "selected_repos": selected_repos,
        },
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        n_critic_runs=args.n_critic_runs,
        critic=create_critic(args),
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

    evaluator = EvoClawEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
    )
    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("EvoClaw inference completed")
    print(json.dumps({"output_json": str(evaluator.output_path)}))


if __name__ == "__main__":
    main()
