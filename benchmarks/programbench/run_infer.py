"""ProgramBench inference script using openhands-sdk as the agent.

ProgramBench (https://github.com/facebookresearch/ProgramBench) gives the
agent a compiled binary plus its public documentation inside a sandboxed
Docker container, and asks the agent to rebuild a working codebase from
scratch. Evaluation (``programbench eval``) then compiles the agent's
submission and runs hidden behavioral tests against the rebuilt binary.

This script handles the inference half of that loop:

1. Load task metadata from the upstream ``programbench`` package.
2. For each instance, layer ``openhands-agent-server`` on top of the
   ``programbench/<id_with_1776>:task_cleanroom`` image and start a
   workspace.
3. Run the OpenHands SDK agent with a prompt asking it to reconstruct the
   codebase under ``/workspace``.
4. Tar up ``/workspace`` into ``<output>/run/<instance_id>/submission.tar.gz``
   in the layout expected by ``programbench eval``.

Usage:
    uv run programbench-infer .llm_config/claude.json --n-limit 5
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, List

from jinja2 import Environment, FileSystemLoader

from benchmarks.programbench.config import INFER_DEFAULTS
from benchmarks.utils.acp import (
    add_acp_agent_metadata,
    build_acp_agent,
    get_acp_forward_env,
    is_acp_agent,
    setup_acp_workspace,
    workspace_keepalive,
)
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
from benchmarks.utils.litellm_proxy import build_eval_llm
from benchmarks.utils.llm_config import load_llm_config
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
)
from openhands.sdk import Agent, Conversation, Tool, get_logger
from openhands.sdk.agent import ACPAgent
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.delegate import DelegateTool
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerDevWorkspace


logger = get_logger(__name__)


# Subdirectory under ``eval_output_dir`` that ``programbench eval`` consumes.
# Each instance lives at ``<eval_output_dir>/run/<instance_id>/submission.tar.gz``.
RUN_SUBDIR = "run"


def _instance_to_image(instance_id: str, tag: str) -> str:
    """Return the upstream cleanroom image name for an instance.

    ProgramBench stores Docker images at
    ``programbench/<owner>_1776_<repo>.<sha>:<tag>``. The ``__`` in the
    instance id (between owner and repo) is replaced with ``_1776_`` so
    Docker tag rules accept it.
    """
    return f"programbench/{instance_id.replace('__', '_1776_')}:{tag}"


def _load_upstream_instances() -> list[dict]:
    """Return the upstream task metadata list.

    Falls back to a clear error if ``programbench`` isn't installed so the
    user gets an actionable message instead of an opaque ImportError.
    """
    try:
        from programbench.utils.load_data import load_all_instances
    except ImportError as exc:
        raise RuntimeError(
            "The 'programbench' package is not installed. Add it to your "
            "environment with `uv pip install programbench` or run "
            "`make build` after pinning a programbench version in "
            "pyproject.toml."
        ) from exc
    # ``include_tests=False`` keeps memory low — the per-task tests.json
    # files are large and only needed during evaluation.
    return load_all_instances(include_tests=False)


def _select_instances(
    all_instances: list[dict],
    selected_file: str | None,
    n_limit: int,
) -> list[dict]:
    """Filter the upstream instance list by ``--select`` and ``--n-limit``."""
    instances = all_instances
    if selected_file:
        with open(selected_file) as fh:
            wanted = {line.strip() for line in fh if line.strip()}
        if not wanted:
            raise ValueError(f"--select file {selected_file!r} is empty")
        instances = [i for i in instances if i["instance_id"] in wanted]
        missing = wanted - {i["instance_id"] for i in instances}
        if missing:
            raise ValueError(
                f"--select listed {len(missing)} unknown instance ids: "
                f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            )
    if n_limit > 0:
        instances = instances[:n_limit]
    return instances


def _render_instruction(
    instance: dict,
    metadata: EvalMetadata,
) -> str:
    """Render the prompt template for a ProgramBench instance."""
    assert metadata.prompt_path is not None
    prompt_path = Path(metadata.prompt_path)
    env = Environment(loader=FileSystemLoader(str(prompt_path.parent)))
    template = env.get_template(prompt_path.name)
    # The cleanroom image conventionally places the binary at
    # ``/workspace/<repo_name>`` (matching the upstream repo basename), but
    # the agent is told to discover it itself — we just provide a hint.
    repo_name = instance["repository"].split("/")[-1]
    return template.render(
        workspace_dir="/workspace",
        binary_path=f"/workspace/{repo_name}",
        language=instance.get("language", "unknown"),
        repository=instance["repository"],
    )


class ProgramBenchEvaluation(Evaluation):
    """ProgramBench evaluation orchestrator built on the shared Evaluation base."""

    def prepare_instances(self) -> List[EvalInstance]:
        upstream = _load_upstream_instances()
        details = self.metadata.details or {}
        selected_file = self.metadata.selected_instances_file
        n_limit = self.metadata.eval_limit

        filtered = _select_instances(upstream, selected_file, n_limit)
        logger.info(
            "Loaded %d ProgramBench tasks (selected=%s, n_limit=%d)",
            len(filtered),
            selected_file or "<all>",
            n_limit,
        )

        task_image_tag = str(
            details.get("task_image_tag") or INFER_DEFAULTS["task_image_tag"]
        )
        instances: list[EvalInstance] = []
        for entry in filtered:
            data = dict(entry)
            data["task_image"] = _instance_to_image(
                entry["instance_id"], task_image_tag
            )
            instances.append(EvalInstance(id=entry["instance_id"], data=data))
        return instances

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """Create a Docker workspace layered on the cleanroom task image.

        ProgramBench's leaderboard rules require the agent to have **no
        internet access** during inference. Achieving that *and* keeping the
        SDK's HTTP control channel alive is non-trivial:

        * ``--network none`` blocks the SDK from reaching the agent-server
          because Docker port mappings need a network interface.
        * ``docker network create --internal`` blocks ``-p`` port mappings.
        * The robust answer is in-container egress filtering (iptables in an
          init step), which needs ``CAP_NET_ADMIN`` and is **future work**.

        For now we leave ``network=None`` (default bridge). The system prompt
        explicitly tells the agent it has no internet, and the cleanroom image
        ships with everything the task needs locally. Agents that try to call
        out anyway will produce non-leaderboard-faithful runs — that limitation
        is documented in the README and tracked in AGENTS.md.
        """
        details = self.metadata.details or {}
        forward_env = get_acp_forward_env(self.metadata.agent_type, forward_env)

        if self.metadata.workspace_type == "docker":
            base_image = instance.data["task_image"]
            target = str(details.get("build_target", INFER_DEFAULTS["build_target"]))
            logger.info(
                "Building agent-server layer on top of %s (target=%s)",
                base_image,
                target,
            )
            workspace = DockerDevWorkspace(
                base_image=base_image,
                working_dir=str(
                    details.get("workspace_dir", INFER_DEFAULTS["workspace_dir"])
                ),
                target=target,  # type: ignore[arg-type]
                forward_env=forward_env or [],
                # See docstring above. Strict offline isolation is follow-up
                # work; today we rely on the prompt + cleanroom image.
                network=None,
            )
        elif self.metadata.workspace_type == "remote":
            raise NotImplementedError(
                "Remote workspace is not yet supported for ProgramBench. "
                "Use --workspace docker for now."
            )
        else:
            raise ValueError(
                f"Unsupported workspace_type for ProgramBench: "
                f"{self.metadata.workspace_type}"
            )

        return workspace

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """Run the agent and collect the submission tarball."""
        if is_acp_agent(self.metadata.agent_type):
            agent = build_acp_agent(self.metadata.agent_type, self.metadata.llm.model)
        else:
            agent_llm = build_eval_llm(self.metadata.llm)
            tools = get_default_tools(enable_browser=False)
            if self.metadata.enable_delegation:
                tools.append(Tool(name=DelegateTool.name))
            condenser = None
            if self.metadata.enable_condenser:
                condenser = LLMSummarizingCondenser(
                    llm=build_eval_llm(self.metadata.llm, usage_id="condenser"),
                    max_size=self.metadata.condenser_max_size,
                    keep_first=self.metadata.condenser_keep_first,
                )
            agent_context = create_agent_context()
            agent = Agent(
                llm=agent_llm,
                tools=tools,
                system_prompt_kwargs={"cli_mode": True},
                agent_context=agent_context,
                condenser=condenser,
            )

        assert isinstance(workspace, RemoteWorkspace)
        setup_acp_workspace(self.metadata.agent_type, workspace)

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

        instruction = _render_instruction(instance.data, self.metadata)
        with workspace_keepalive(self.metadata.agent_type, workspace):
            conversation.send_message(instruction)
            run_timeout = int(os.getenv("CONVERSATION_TIMEOUT", "7200"))
            conversation.run(timeout=run_timeout)

        history = list(conversation.state.events)
        submission_path = self._collect_submission(instance, workspace)
        logger.info("Wrote submission tarball: %s", submission_path)

        summarize_instance(
            instance_id=instance.id,
            conversation=conversation,
            git_patch="",  # Programbench has no git patch; the submission is the artifact.
            logger=logger,
        )

        test_result: dict[str, Any] = {
            "submission_path": str(submission_path),
            "submission_size_bytes": submission_path.stat().st_size,
            "task_image": instance.data["task_image"],
        }
        if isinstance(agent, ACPAgent):
            add_acp_agent_metadata(test_result, conversation)

        return EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result=test_result,
            instruction=instruction,
            error=None,
            history=history,
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )

    def _collect_submission(
        self,
        instance: EvalInstance,
        workspace: RemoteWorkspace,
    ) -> Path:
        """Tar up ``/workspace`` from the container into submission.tar.gz.

        The output layout matches the one ``programbench eval`` expects::

            <eval_output_dir>/run/<instance_id>/submission.tar.gz
        """
        details = self.metadata.details or {}
        workspace_dir = str(
            details.get("workspace_dir", INFER_DEFAULTS["workspace_dir"])
        )

        run_dir = Path(self.metadata.eval_output_dir) / RUN_SUBDIR / instance.id
        run_dir.mkdir(parents=True, exist_ok=True)
        submission_path = run_dir / "submission.tar.gz"

        # Build the tar inside the container so we don't need to stream the
        # whole tree over HTTP, then download a single archive. We exclude
        # the reference binary itself (per the prompt: "do not modify or
        # move the reference binary") since it could otherwise be tarred
        # back into the submission and short-circuit the eval. The eval
        # harness ignores extra files it doesn't recognize.
        repo_basename = instance.data["repository"].split("/")[-1]
        in_container_tar = "/tmp/submission.tar.gz"
        tar_cmd = (
            f"cd {shlex.quote(workspace_dir)} && "
            f"tar --exclude={shlex.quote(repo_basename)} "
            f"-czf {in_container_tar} ."
        )
        result = workspace.execute_command(tar_cmd, timeout=600)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to create submission tarball for {instance.id}: "
                f"exit_code={result.exit_code} stderr={result.stderr!r}"
            )

        # Pull the tarball out of the container. We try the workspace's
        # download_directory hook first (used by some workspace impls),
        # then fall back to ``cat``ing the file out.
        download_directory = getattr(workspace, "download_directory", None)
        if download_directory is not None:
            try:
                tmp = download_directory(in_container_tar)
                if tmp and Path(tmp).exists():
                    shutil.move(tmp, submission_path)
                    return submission_path
            except Exception as exc:  # pragma: no cover - workspace impl detail
                logger.warning(
                    "download_directory failed for %s: %s; falling back to cat",
                    instance.id,
                    exc,
                )

        # Fallback: stream the tar bytes through ``base64`` to avoid mangling
        # binary content over the workspace HTTP API. This is slower but
        # works against any RemoteWorkspace.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".b64", delete=False
        ) as encoded_tmp:
            encoded_path = encoded_tmp.name
        try:
            cat_cmd = f"base64 -w0 {in_container_tar}"
            cat_result = workspace.execute_command(cat_cmd, timeout=600)
            if cat_result.exit_code != 0:
                raise RuntimeError(
                    f"Failed to read submission tarball for {instance.id}: "
                    f"exit_code={cat_result.exit_code} stderr={cat_result.stderr!r}"
                )
            submission_path.write_bytes(base64.b64decode(cat_result.stdout))
        finally:
            Path(encoded_path).unlink(missing_ok=True)

        return submission_path


def _validate_offline_constraint(workspace_type: str, args_offline: bool) -> None:
    """Surface ProgramBench's offline-agent invariant to the user.

    The ProgramBench paper forbids the agent from having internet access
    during inference. We currently rely on the prompt + cleanroom image to
    keep the agent honest; strict in-container egress filtering (iptables
    with ``CAP_NET_ADMIN``) is a known follow-up. Always log the caveat so
    runs are not mistaken for leaderboard-faithful submissions.
    """
    logger.warning(
        "ProgramBench: strict offline isolation is not yet enforced inside "
        "the agent container. Runs may not be leaderboard-faithful unless the "
        "agent stays offline of its own accord. (--allow-network=%s, "
        "workspace=%s)",
        not args_offline,
        workspace_type,
    )
    if workspace_type != "docker":
        logger.warning(
            "Network isolation only applies to --workspace docker; remote "
            "workspaces have no guarantees yet."
        )


def _create_submission_layout(eval_output_dir: str) -> Path:
    """Ensure ``<eval_output_dir>/run`` exists and return its path."""
    run_dir = Path(eval_output_dir) / RUN_SUBDIR
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    parser = get_parser()
    add_prompt_path_argument(parser, __file__)
    parser.add_argument(
        "--task-image-tag",
        type=str,
        help="Tag of the upstream task image to mount as the agent's base "
        "image (default: task_cleanroom).",
    )
    parser.add_argument(
        "--build-target",
        type=str,
        choices=["binary", "binary-minimal", "source", "source-minimal"],
        help="Agent-server build target layered on top of the cleanroom image.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        default=False,
        help="Reserved for future strict-offline mode. Today the agent "
        "container always has internet via the default Docker bridge "
        "(see prepare_workspace docstring). This flag is recorded in "
        "metadata so runs remain reproducible once strict isolation lands.",
    )
    parser.set_defaults(**INFER_DEFAULTS)
    args = parser.parse_args()

    if args.n_critic_runs < 1:
        raise ValueError(f"n_critic_runs must be >= 1, got {args.n_critic_runs}")

    llm = load_llm_config(args.llm_config_path)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    _validate_offline_constraint(args.workspace, not args.allow_network)

    dataset_description = (
        str(args.dataset).replace("/", "__") + "-" + str(args.split).replace("/", "__")
    )
    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )
    _create_submission_layout(structured_output_dir)

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={
            "task_image_tag": args.task_image_tag,
            "build_target": args.build_target,
            "workspace_dir": str(INFER_DEFAULTS["workspace_dir"]),
            "offline_inference": not args.allow_network,
        },
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=None,
        n_critic_runs=args.n_critic_runs,
        critic=create_critic(args),
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        enable_delegation=args.enable_delegation,
        agent_type=args.agent_type,
    )

    evaluator = ProgramBenchEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
    )
    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("ProgramBench inference completed!")
    print(json.dumps({"output_json": str(evaluator.output_path)}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
