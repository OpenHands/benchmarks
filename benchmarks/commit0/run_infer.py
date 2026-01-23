import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, List

from commit0.harness.constants import SPLIT
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader

from benchmarks.commit0.build_images import (
    extract_custom_tag,
    get_base_docker_image,
)
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.conversation import build_event_persistence_callback
from benchmarks.utils.critics import create_critic
from benchmarks.utils.dataset import prepare_dataset
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
from openhands.sdk import LLM, Agent, Conversation, get_logger
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import APIRemoteWorkspace, DockerDevWorkspace


logger = get_logger(__name__)


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
    
    Note:
        The commit0-lite benchmark contains 16 instances total, but only 10 are used
        as reference (gold) instances for accuracy calculation on the leaderboard.
        
        Issue: PR #351 showed 100.7% accuracy because we were including all 16 repos
        instead of just the 10 reference repos, leading to incorrect test totals.
        
        References:
        - Leaderboard: https://commit-0.github.io/analysis/
        - Breakdown: https://commit-0.github.io/analysis_commit0_lite_reference/
        
        The 10 reference repos are: simpy, tinydb, marshmallow, wcwidth, imapclient,
        voluptuous, jinja, deprecated, cookiecutter, cachetools
    """
    import pandas as pd

    # The 10 reference (gold) repos used for leaderboard scoring
    # See: https://commit-0.github.io/analysis_commit0_lite_reference/
    REFERENCE_REPOS = {
        "simpy",
        "tinydb",
        "marshmallow",
        "wcwidth",
        "imapclient",
        "voluptuous",
        "jinja",
        "deprecated",
        "cookiecutter",
        "cachetools",
    }

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

    # Filter to only reference repos for accuracy calculation
    # This prevents including non-reference repos like chardet, pyjwt, etc.
    filtered_dataset = filtered_dataset[
        filtered_dataset["instance_id"].isin(REFERENCE_REPOS)
    ]
    
    logger.info(
        f"Filtered to {len(filtered_dataset)} reference instances from {len(REFERENCE_REPOS)} reference repos"
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
        repo_split: str = "lite",
        dataset_name: str = "wentingzhao/commit0_combined",
        dataset_split: str = "test",
    ):
        super().__init__(metadata=metadata, num_workers=num_workers)
        # Store additional parameters in metadata.details for access in methods
        if not hasattr(metadata, "details") or metadata.details is None:
            metadata.details = {}
        metadata.details.update(
            {
                "repo_split": repo_split,
                "dataset_name": dataset_name,
                "dataset_split": dataset_split,
            }
        )

    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up Commit0 evaluation data")

        details = self.metadata.details or {}
        dataset_name = details.get("dataset_name", "wentingzhao/commit0_combined")
        dataset_split = details.get("dataset_split", "test")
        repo_split = details.get("repo_split", "lite")

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
        repo_name = instance.data["repo"].split("/")[1]
        base_docker_image = get_base_docker_image(repo_name)
        build_target = "source-minimal"
        logger.info(f"Using base docker image: {base_docker_image}")

        if self.metadata.workspace_type == "docker":
            # Build agent-server image from base commit0 image
            workspace = DockerDevWorkspace(
                base_image=base_docker_image,
                working_dir="/workspace",
                target=build_target,
                forward_env=forward_env or [],
            )
            logger.info(
                f"Building workspace from {base_docker_image}. This may take a while..."
            )
        elif self.metadata.workspace_type == "remote":
            runtime_api_key = os.getenv("RUNTIME_API_KEY")
            if not runtime_api_key:
                raise ValueError(
                    "RUNTIME_API_KEY environment variable is not set for remote workspace"
                )

            sdk_short_sha = os.getenv("SDK_SHORT_SHA", SDK_SHORT_SHA)
            custom_tag = extract_custom_tag(base_docker_image)
            suffix = f"-{build_target}" if build_target != "binary" else ""
            agent_server_image = (
                f"{EVAL_AGENT_SERVER_IMAGE}:{sdk_short_sha}-{custom_tag}{suffix}"
            )

            if not image_exists(agent_server_image):
                raise RuntimeError(
                    f"Agent server image {agent_server_image} does not exist in container registry. "
                    "Run 'benchmarks/commit0/build_images.py --push' to build and push it first."
                )

            logger.info(
                f"Using remote workspace with image {agent_server_image} "
                f"(sdk sha: {sdk_short_sha}, resource_factor: {resource_factor})"
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
        workspace_dir_name = instance.data["repo"].split("/")[1]
        clone_cmd = f"cd /workspace/ && git clone -b commit0_combined https://github.com/{instance.data['repo']}.git {workspace_dir_name}"
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

        tools = get_default_tools(enable_browser=False)
        agent = Agent(
            llm=self.metadata.llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
        )

        assert isinstance(workspace, RemoteWorkspace)

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
        )

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
        )
        conversation.send_message(instruction)
        run_timeout = int(os.getenv("CONVERSATION_TIMEOUT", "3600"))
        conversation.run(timeout=run_timeout)

        history = list(conversation.state.events)

        # Complete runtime: git add, commit, diff, run tests
        workspace.execute_command(f"cd {repo_path} && git add .", timeout=600)
        workspace.execute_command(
            f"cd {repo_path} && "
            'git config --global user.email "evaluation@openhands.dev" && '
            'git config --global user.name "OpenHands Evaluation" && '
            'git commit -m "openhands edits"',
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
        # Use python -m pytest instead of pytest command to avoid permission issues
        if test_cmd.strip() == "pytest":
            test_cmd = "python -m pytest"
        full_test_cmd = f"cd {repo_path} && {test_cmd} --json-report --json-report-file=report.json --continue-on-collection-errors {test_dir} > test_output.txt 2>&1"
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

        # Read test report
        report_result = workspace.execute_command(
            f"cd {repo_path} && cat report.json",
            timeout=600,
        )

        # Debug logging for report
        logger.info(f"Report read exit code: {report_result.exit_code}")
        if report_result.exit_code == 0:
            logger.info(f"Report content length: {len(report_result.stdout)}")
            logger.info(
                f"Report preview: {report_result.stdout[:200]}..."
            )  # First 200 chars
        else:
            logger.info(f"Failed to read report.json: {report_result.stderr}")
            # Check if file exists
            check_file = workspace.execute_command(
                f"cd {repo_path} && ls -la report.json", timeout=60
            )
            logger.info(
                f"File check: {check_file.stdout if check_file.exit_code == 0 else check_file.stderr}"
            )

        # Initialize eval_result with default values
        eval_result = {
            "name": workspace_dir_name,
            "sum": 0,
            "passed": 0,
            "num_passed": 0,
            "num_tests": len(test_ids),
        }

        if report_result.exit_code == 0:
            try:
                report = json.loads(report_result.stdout.strip())
                logger.info(
                    f"JSON report parsed successfully. Keys: {list(report.keys())}"
                )
                if "tests" in report:
                    logger.info(f"Found {len(report['tests'])} test entries in report")
                else:
                    logger.warning("No 'tests' key found in report")
                tests = {x["nodeid"]: x["call"] for x in report["tests"] if "call" in x}
                logger.info(f"Extracted {len(tests)} tests with 'call' data")

                # If test_ids is empty (commit0 get-tests failed), use test IDs from JSON report
                if not test_ids and tests:
                    test_ids = list(tests.keys())
                    logger.info(
                        f"Using test IDs from JSON report: {len(test_ids)} tests"
                    )

                status = []
                runtimes = []
                no_runs = 0

                for test_id in test_ids:
                    if test_id in tests and tests[test_id] is not None:
                        status.append(tests[test_id]["outcome"])
                        runtimes.append(tests[test_id]["duration"])
                        no_runs += 1
                    else:
                        status.append("failed")
                        runtimes.append(0)

                status_counts = Counter(status)
                total_runtime = sum(runtimes) if no_runs > 0 else 0
                num_passed = status_counts.get("passed", 0) + status_counts.get(
                    "xfail", 0
                )
                passed_ratio = num_passed / len(status) if status else 0

                # Debug logging for final calculations
                logger.info(f"Status counts: {dict(status_counts)}")
                logger.info(
                    f"Total runtime: {total_runtime}, Num passed: {num_passed}, Passed ratio: {passed_ratio}"
                )
                logger.info(
                    f"Total test IDs: {len(test_ids)}, Status list length: {len(status)}"
                )

                eval_result = {
                    "name": workspace_dir_name,
                    "sum": total_runtime,
                    "passed": passed_ratio,
                    "num_passed": num_passed,
                    "num_tests": len(test_ids),
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse test report JSON: {e}")
                logger.error(
                    f"Raw JSON content: {report_result.stdout[:500]}..."
                )  # First 500 chars
                # eval_result already has default values, no need to reassign
        else:
            logger.warning(
                f"Report reading failed with exit code {report_result.exit_code}"
            )
            logger.warning(f"Report stderr: {report_result.stderr}")

        # Final debug log
        logger.info(f"Final eval_result: {eval_result}")

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

        test_result = {
            "eval_result": eval_result,
        }

        out = EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result=test_result,
            instruction=instruction,
            error=None,
            history=history,
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
        "--repo-split",
        type=str,
        default="lite",
        help="all, lite, or each repo name",
    )
    # Override the default dataset for commit0
    parser.set_defaults(dataset="wentingzhao/commit0_combined")
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
        max_attempts=args.max_attempts,
        critic=create_critic(args),
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
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


if __name__ == "__main__":
    main()
