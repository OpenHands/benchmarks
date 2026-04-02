import json
import os
import re
import shlex
from typing import Any, List

from commit0.harness.constants import SPLIT
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader

from benchmarks.commit0.build_images import (
    extract_custom_tag,
    get_base_docker_image,
)
from benchmarks.commit0.config import INFER_DEFAULTS
from benchmarks.utils.acp import (
    add_acp_agent_metadata,
    build_acp_agent,
    get_acp_forward_env,
    is_acp_agent,
    setup_acp_workspace,
    workspace_keepalive,
)
from benchmarks.utils.args_parser import add_prompt_path_argument, get_parser
from benchmarks.utils.console_logging import summarize_instance
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.conversation import build_event_persistence_callback
from benchmarks.utils.critics import create_critic
from benchmarks.utils.dataset import prepare_dataset
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.image_utils import create_docker_workspace, remote_image_exists
from benchmarks.utils.llm_config import load_llm_config
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
)
from benchmarks.utils.version import IMAGE_TAG_PREFIX
from openhands.sdk import Agent, Conversation, Tool, get_logger
from openhands.sdk.context import AgentContext
from openhands.sdk.context.skills.skill import load_public_skills
from openhands.sdk.agent import ACPAgent
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.delegate import DelegateTool
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import APIRemoteWorkspace


logger = get_logger(__name__)

# Script run inside the container to extract summary + duration from
# report.json.  Only the small summary dict (~200 bytes) is printed to
# stdout, avoiding HTTP transfer corruption for large reports (see #511).
EXTRACT_SUMMARY_SCRIPT = """
import json
r = json.load(open('report.json'))
s = r.get('summary', {})
s['duration'] = r.get('duration', 0)
print(json.dumps(s))
""".strip()


def normalize_pytest_cmd(test_cmd: str) -> str:
    """Replace bare pytest/pytest3 with python -m pytest to avoid PATH/permission issues."""
    if (
        re.match(r"pytest\d?(\s|$)", test_cmd.strip())
        and "python -m pytest" not in test_cmd
    ):
        test_cmd = re.sub(r"\bpytest(\d?)", r"python -m pytest\1", test_cmd, count=1)
    return test_cmd


def get_pythonpath_prefix(src_dir: str) -> str:
    """Return PYTHONPATH env prefix for src-layout repos."""
    if src_dir and src_dir.startswith("src"):
        return "PYTHONPATH=src:$PYTHONPATH "
    return ""


def parse_report_summary(raw_json: str) -> dict:
    """Parse pytest-json-report summary extracted from the container.

    Expects the JSON output of the in-container extraction command, which
    produces the ``summary`` dict from pytest-json-report (v1.5.0) with
    ``duration`` injected from the top-level report field::

        {"passed": 100, "failed": 5, "total": 105, "collected": 105, "duration": 45.2}

    Args:
        raw_json: JSON string with the summary dict (including injected 'duration').

    Returns:
        Dict with keys: sum, passed, num_passed, num_tests.

    Raises:
        json.JSONDecodeError: If raw_json is not valid JSON.
        ValueError: If the summary is missing or has an empty 'total' field.
    """
    summary = json.loads(raw_json.strip())

    if "total" not in summary or summary["total"] == 0:
        raise ValueError(f"Report summary missing or empty 'total' field: {summary}")

    num_passed = summary.get("passed", 0) + summary.get("xfailed", 0)
    num_tests = summary["total"]
    total_runtime = summary.get("duration", 0)
    passed_ratio = num_passed / num_tests

    return {
        "sum": total_runtime,
        "passed": passed_ratio,
        "num_passed": num_passed,
        "num_tests": num_tests,
    }


def get_instruction(
    instance: dict,
    metadata: EvalMetadata,
) -> str:
    """Generate instruction for the agent."""
    workspace_dir_name = instance["repo"].split("/")[1]
    test_cmd = instance["test"]["test_cmd"]
    test_dir = instance["test"]["test_dir"]

    assert metadata.prompt_path is not None
    prompts_dir = os.path.dirname(metadata.prompt_path)
    template_name = os.path.basename(metadata.prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    context = {
        "workspace_dir_name": workspace_dir_name,
        "test_cmd": test_cmd,
        "test_dir": test_dir,
    }

    instruction = template.render(context)
    return instruction


def commit0_setup(df: Any, repo_split: str) -> Any:
    """Setup Commit0 dataset based on split type.

    Args:
        df: Full Commit0 dataset (pandas DataFrame)
        repo_split: Split type ('all', 'lite' or specific repo name)

    Returns:
        Filtered dataset based on split type
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()

    filtered_dataset = pd.concat(
        [
            df[pd.Series(df["repo"]).str.split("/").str[1] == repo]
            for repo in SPLIT.get(repo_split, [])
        ]
    )

    if "setup" in filtered_dataset.columns:
        filtered_dataset = filtered_dataset.drop("setup", axis=1)

    filtered_dataset["instance_id"] = (
        pd.Series(filtered_dataset["repo"]).str.split("/").str[1]
    )

    return filtered_dataset


class Commit0Evaluation(Evaluation):
    """
    Process-based Commit0 evaluation implemented as a child of the
    abstract Evaluation orchestrator.

    Implements:
      - prepare_instances()
      - prepare_workspace(instance)
      - evaluate_instance(instance, workspace)
    """

    def __init__(
        self,
        metadata: EvalMetadata,
        num_workers: int = 1,
        repo_split: str | None = None,
        dataset_name: str | None = None,
        dataset_split: str | None = None,
    ):
        super().__init__(metadata=metadata, num_workers=num_workers)
        # Store additional parameters in metadata.details for access in methods
        if not hasattr(metadata, "details") or metadata.details is None:
            metadata.details = {}
        metadata.details.update(
            {
                "repo_split": repo_split or INFER_DEFAULTS["repo_split"],
                "dataset_name": dataset_name or INFER_DEFAULTS["dataset"],
                "dataset_split": dataset_split or INFER_DEFAULTS["split"],
            }
        )

    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up Commit0 evaluation data")

        details = self.metadata.details or {}
        dataset_name = details.get("dataset_name", INFER_DEFAULTS["dataset"])
        dataset_split = details.get("dataset_split", INFER_DEFAULTS["split"])
        repo_split = details.get("repo_split", INFER_DEFAULTS["repo_split"])

        dataset = load_dataset(dataset_name, split=dataset_split)
        df = commit0_setup(dataset, repo_split)

        if self.metadata.selected_instances_file:
            df = prepare_dataset(
                dataset=df,
                n_limit=None,
                selected_instances_file=self.metadata.selected_instances_file,
            )
            self.metadata.eval_limit = len(df)

        instances: List[EvalInstance] = []
        for _, row in df.iterrows():
            inst_id = str(row["instance_id"])
            instances.append(EvalInstance(id=inst_id, data=row.to_dict()))

        # Apply eval_limit if specified
        if self.metadata.eval_limit > 0:
            instances = instances[: self.metadata.eval_limit]
            logger.info(
                "Limited instances to %d (eval_limit=%d)",
                len(instances),
                self.metadata.eval_limit,
            )

        logger.info("Total instances to process: %d", len(instances))
        return instances

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """
        Create workspace and set up the commit0 repository.

        Args:
            instance: The evaluation instance to prepare workspace for.
            resource_factor: Resource factor for runtime allocation (default: 1).
                           Higher values allocate more CPU/memory resources.
                           Used by APIRemoteWorkspace for remote runtime allocation.
            forward_env: Environment variables to forward into the workspace.
        """
        forward_env = get_acp_forward_env(self.metadata.agent_type, forward_env)

        repo_name = instance.data["repo"].split("/")[1]
        base_docker_image = get_base_docker_image(repo_name)
        build_target = "source-minimal"
        logger.info(f"Using base docker image: {base_docker_image}")

        if self.metadata.workspace_type == "docker":
            custom_tag = extract_custom_tag(base_docker_image)
            suffix = f"-{build_target}" if build_target != "binary" else ""
            agent_server_image = (
                f"{EVAL_AGENT_SERVER_IMAGE}:{IMAGE_TAG_PREFIX}-{custom_tag}{suffix}"
            )
            workspace = create_docker_workspace(
                agent_server_image=agent_server_image,
                base_image=base_docker_image,
                build_target=build_target,
                forward_env=forward_env,
            )
        elif self.metadata.workspace_type == "remote":
            runtime_api_key = os.getenv("RUNTIME_API_KEY")
            if not runtime_api_key:
                raise ValueError(
                    "RUNTIME_API_KEY environment variable is not set for remote workspace"
                )

            custom_tag = extract_custom_tag(base_docker_image)
            suffix = f"-{build_target}" if build_target != "binary" else ""
            agent_server_image = (
                f"{EVAL_AGENT_SERVER_IMAGE}:{IMAGE_TAG_PREFIX}-{custom_tag}{suffix}"
            )

            if not remote_image_exists(agent_server_image):
                raise RuntimeError(
                    f"Agent server image {agent_server_image} does not exist in container registry. "
                    "Run 'benchmarks/commit0/build_images.py --push' to build and push it first."
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

        # Clone the repository to the specific directory
        # Use --depth 1 for shallow clone to prevent agents from accessing git history
        # and exploiting it to retrieve original implementations (reward hacking prevention)
        workspace_dir_name = instance.data["repo"].split("/")[1]
        clone_cmd = f"cd /workspace/ && git clone --depth 1 -b commit0_combined https://github.com/{instance.data['repo']}.git {workspace_dir_name}"
        res = workspace.execute_command(clone_cmd, timeout=600)
        if res.exit_code != 0:
            raise RuntimeError(f"Failed to clone repo: {res.stderr}")
        logger.info(f"Cloned repository: {instance.data['repo']}")

        # Create new branch
        branch_cmd = f"cd /workspace/{workspace_dir_name} && git checkout -b openhands"
        res = workspace.execute_command(branch_cmd, timeout=600)
        if res.exit_code != 0:
            raise RuntimeError(f"Failed to create branch: {res.stderr}")
        logger.info("Created new branch: openhands")

        # Install commit0
        # Try uv first, fall back to pip if uv is not available
        install_cmd = f"cd /workspace/{workspace_dir_name} && (uv pip install commit0 || pip install commit0)"
        res = workspace.execute_command(install_cmd, timeout=600)
        if res.exit_code != 0:
            raise RuntimeError(f"Failed to install commit0: {res.stderr}")
        logger.info("Installed commit0")

        # Install pytest and required plugins for test reporting
        plugin_install_cmd = f"cd /workspace/{workspace_dir_name} && (uv pip install pytest pytest-json-report pytest-cov || pip install pytest pytest-json-report pytest-cov)"
        res = workspace.execute_command(plugin_install_cmd, timeout=600)
        if res.exit_code != 0:
            raise RuntimeError(f"Failed to install pytest and plugins: {res.stderr}")
        logger.info("Installed pytest and required plugins")

        # Verify pytest and plugin installation
        verify_pytest_cmd = (
            f"cd /workspace/{workspace_dir_name} && python -m pytest --version"
        )
        verify_pytest_res = workspace.execute_command(verify_pytest_cmd, timeout=60)
        logger.info(f"Pytest verification exit code: {verify_pytest_res.exit_code}")
        if verify_pytest_res.exit_code == 0:
            logger.info(f"Pytest available: {verify_pytest_res.stdout.strip()}")
        else:
            logger.warning(f"Pytest verification failed: {verify_pytest_res.stderr}")

        verify_plugin_cmd = f"cd /workspace/{workspace_dir_name} && python -c 'import pytest_jsonreport; print(\"Plugin available\")'"
        verify_plugin_res = workspace.execute_command(verify_plugin_cmd, timeout=60)
        logger.info(f"Plugin verification exit code: {verify_plugin_res.exit_code}")
        if verify_plugin_res.exit_code == 0:
            logger.info("pytest-json-report plugin verified successfully")
        else:
            logger.warning(f"Plugin verification failed: {verify_plugin_res.stderr}")

        return workspace

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """
        Run agent, collect history, git patch, and test results.
        """
        workspace_dir_name = instance.data["repo"].split("/")[1]
        repo_path = f"/workspace/{workspace_dir_name}"

        if is_acp_agent(self.metadata.agent_type):
            agent = build_acp_agent(self.metadata.agent_type, self.metadata.llm.model)
        else:
            tools = get_default_tools(enable_browser=False)
            if self.metadata.enable_delegation:
                tools.append(Tool(name=DelegateTool.name))
            condenser = None
            if self.metadata.enable_condenser:
                condenser = LLMSummarizingCondenser(
                    llm=self.metadata.llm.model_copy(update={"usage_id": "condenser"}),
                    max_size=self.metadata.condenser_max_size,
                    keep_first=self.metadata.condenser_keep_first,
                )
            # Load public skills (respects EXTENSIONS_REF env var)

            skills = load_public_skills()

            agent_context = AgentContext(skills=skills) if skills else None

            

            agent = Agent(

                llm=self.metadata.llm,

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

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
        )
        with workspace_keepalive(self.metadata.agent_type, workspace):
            conversation.send_message(instruction)
            run_timeout = int(os.getenv("CONVERSATION_TIMEOUT", "3600"))
            conversation.run(timeout=run_timeout)

        history = list(conversation.state.events)

        # Complete runtime: git add, commit, diff, run tests
        workspace.execute_command(f"cd {repo_path} && git add .", timeout=600)
        # Use --no-verify to bypass pre-commit hooks (e.g., husky) that can fail
        workspace.execute_command(
            f"cd {repo_path} && "
            'git config --global user.email "evaluation@openhands.dev" && '
            'git config --global user.name "OpenHands Evaluation" && '
            'git commit --no-verify -m "openhands edits"',
            timeout=600,
        )

        # Get git patch
        base_commit = instance.data["base_commit"]
        git_patch = None
        for retry in range(5):
            patch_result = workspace.execute_command(
                f"cd {repo_path} && git diff {base_commit} HEAD -- . ':(exclude)spec.pdf.bz2'",
                timeout=600 + 100 * retry,
            )
            if patch_result.exit_code == 0:
                git_patch = patch_result.stdout.strip()
                break
            logger.info("Failed to get git diff, retrying...")

        if git_patch is None:
            raise RuntimeError("Failed to get git patch after 5 retries")

        # Run tests
        test_cmd = instance.data["test"]["test_cmd"]
        test_dir = instance.data["test"]["test_dir"]
        test_cmd = normalize_pytest_cmd(test_cmd)
        src_dir = instance.data.get("src_dir", "")
        env_prefix = get_pythonpath_prefix(src_dir)
        full_test_cmd = f"cd {repo_path} && {env_prefix}{test_cmd} --json-report --json-report-file=report.json --continue-on-collection-errors {test_dir} > test_output.txt 2>&1"
        logger.info(f"Running test command: {full_test_cmd}")
        test_result = workspace.execute_command(full_test_cmd, timeout=600)
        logger.info(f"Test command exit code: {test_result.exit_code}")
        if test_result.exit_code != 0:
            logger.warning(f"Test command failed with stderr: {test_result.stderr}")
            logger.warning(f"Test command failed with stdout: {test_result.stdout}")

        # Read test output
        test_output_result = workspace.execute_command(
            f"cd {repo_path} && cat test_output.txt",
            timeout=600,
        )
        test_output = (
            test_output_result.stdout.strip()
            if test_output_result.exit_code == 0
            else ""
        )

        # Get pytest exit code from the test_result
        pytest_exit_code = str(test_result.exit_code)

        # Get test IDs and parse report
        repo_name = instance.data["repo"].split("/")[1]
        repo_name_normalized = repo_name.replace(".", "-")
        test_ids_result = workspace.execute_command(
            f"cd {repo_path} && commit0 get-tests {repo_name_normalized}",
            timeout=600,
        )
        test_ids = (
            test_ids_result.stdout.strip().split("\n")
            if test_ids_result.exit_code == 0
            else []
        )

        # Debug logging
        logger.info(f"Test IDs command exit code: {test_ids_result.exit_code}")
        logger.info(
            f"Test IDs found: {len(test_ids)} - {test_ids[:3] if test_ids else 'None'}"
        )  # Show first 3

        # Extract summary and duration from report.json inside the container.
        # This avoids transferring the full report (can be >1 MB) over HTTP,
        # which causes corruption for large files (see #511).
        summary_cmd = (
            f"cd {repo_path} && python3 -c {shlex.quote(EXTRACT_SUMMARY_SCRIPT)}"
        )
        report_result = workspace.execute_command(summary_cmd, timeout=600)
        logger.info(f"Report summary extraction exit code: {report_result.exit_code}")

        # Intentionally let errors propagate here — do NOT add a try/except.
        # Silent failures caused instances to be scored 0/0 even when all
        # tests passed (see #511).  The framework's retry logic in
        # evaluation.py handles the exception.
        if report_result.exit_code != 0:
            raise RuntimeError(
                f"Report summary extraction failed (exit code {report_result.exit_code}): "
                f"{report_result.stderr}"
            )

        parsed = parse_report_summary(report_result.stdout)
        logger.info(f"Parsed report summary: {parsed}")

        eval_result = {
            "name": workspace_dir_name,
            **parsed,
        }

        # Final debug log
        logger.info(f"Final eval_result: {eval_result}")

        # Log instance summary
        summarize_instance(
            instance_id=instance.id,
            conversation=conversation,
            git_patch=git_patch or "",
            logger=logger,
        )

        # Save workspace as zip (if supported by workspace implementation)
        zip_dest = os.path.join(
            self.metadata.eval_output_dir, "repos", repo_name, f"{repo_name}.zip"
        )
        os.makedirs(os.path.dirname(zip_dest), exist_ok=True)

        # Try to copy workspace directory if the method is available
        try:
            download_directory = getattr(workspace, "download_directory", None)
            if download_directory is not None:
                temp_zip = download_directory(repo_path)
                if temp_zip and os.path.exists(temp_zip):
                    import shutil

                    shutil.move(temp_zip, zip_dest)
            else:
                logger.warning(
                    "Workspace does not support downloading directory, skipping zip creation"
                )
        except Exception as e:
            logger.warning(f"Failed to save workspace as zip: {e}")

        # Save patch, test output, and exit code
        patch_file = os.path.join(
            self.metadata.eval_output_dir, "repos", repo_name, f"{repo_name}_patch.diff"
        )
        test_output_file = os.path.join(
            self.metadata.eval_output_dir,
            "repos",
            repo_name,
            f"{repo_name}_test_output.txt",
        )
        pytest_exit_code_file = os.path.join(
            self.metadata.eval_output_dir,
            "repos",
            repo_name,
            f"{repo_name}_pytest_exit_code.txt",
        )

        write_targets = [
            (patch_file, git_patch),
            (test_output_file, test_output),
            (pytest_exit_code_file, pytest_exit_code),
        ]

        for write_target in write_targets:
            with open(write_target[0], "w") as f:
                f.write(write_target[1])

        logger.info(
            f"Got evaluation result for repo {instance.id}:\n--------\n{eval_result}\n--------"
        )

        output_test_result: dict[str, Any] = {
            "eval_result": eval_result,
        }
        if isinstance(agent, ACPAgent):
            add_acp_agent_metadata(output_test_result, conversation)

        out = EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result=output_test_result,
            instruction=instruction,
            error=None,
            history=history,
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )
        return out


def main() -> None:
    parser = get_parser()
    add_prompt_path_argument(parser, __file__)
    parser.add_argument(
        "--repo-split",
        type=str,
        help="all, lite, or each repo name",
    )
    # Apply INFER_DEFAULTS from config (matches evaluation repository values.yaml)
    parser.set_defaults(**INFER_DEFAULTS)
    args = parser.parse_args()

    # Validate n_critic_runs
    if args.n_critic_runs < 1:
        raise ValueError(f"n_critic_runs must be >= 1, got {args.n_critic_runs}")

    llm = load_llm_config(args.llm_config_path)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    dataset_description = (
        args.dataset.replace("/", "__") + "-" + args.repo_split.replace("/", "__")
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
        env_setup_commands=None,
        n_critic_runs=args.n_critic_runs,
        critic=create_critic(args),
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        enable_delegation=args.enable_delegation,
        agent_type=args.agent_type,
    )

    evaluator = Commit0Evaluation(
        metadata=metadata,
        num_workers=args.num_workers,
        repo_split=args.repo_split,
        dataset_name=args.dataset,
        dataset_split=args.split,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")
    print(json.dumps({"output_json": str(evaluator.output_path)}))


if __name__ == "__main__":
    main()
