import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, List

from commit0.harness.constants import SPLIT
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader

from benchmarks.utils.args_parser import get_parser
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
from openhands.sdk import LLM, Agent, Conversation, get_logger
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerWorkspace


logger = get_logger(__name__)


def get_docker_image(
    repo_name: str, docker_image_prefix: str = "docker.io/wentingzhao/"
) -> str:
    """Get docker image for a commit0 repository."""
    return (docker_image_prefix.rstrip("/") + "/" + repo_name).lower() + ":v0"


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
            df[df["repo"].str.split("/").str[1] == repo]
            for repo in SPLIT.get(repo_split, [])
        ]
    )

    if "setup" in filtered_dataset.columns:
        filtered_dataset = filtered_dataset.drop("setup", axis=1)

    filtered_dataset["instance_id"] = filtered_dataset["repo"].str.split("/").str[1]  # type: ignore[attr-defined]

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

    def prepare_workspace(self, instance: EvalInstance) -> RemoteWorkspace:
        """
        Create workspace and set up the commit0 repository.
        """
        repo_name = instance.data["repo"].split("/")[1]
        base_docker_image = get_docker_image(repo_name)
        logger.info(f"Using base docker image: {base_docker_image}")

        # Build agent-server image from base commit0 image
        workspace = DockerWorkspace(
            base_image=base_docker_image,
            working_dir="/workspace",
            target="source-minimal",
        )
        logger.info(
            f"Building workspace from {base_docker_image}. This may take a while..."
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

        def _log_event(ev):
            logger.debug("Event: %s", ev)

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[_log_event],
            max_iteration_per_run=self.metadata.max_iterations,
        )

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
        )
        conversation.send_message(instruction)
        conversation.run()

        history = list(map(lambda event: event.model_dump(), conversation.state.events))

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
        test_result = workspace.execute_command(
            f"cd {repo_path} && {test_cmd} --json-report --json-report-file=report.json --continue-on-collection-errors {test_dir} > test_output.txt 2>&1",
            timeout=600,
        )

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

        # Read test report
        report_result = workspace.execute_command(
            f"cd {repo_path} && cat report.json",
            timeout=600,
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
                tests = {x["nodeid"]: x["call"] for x in report["tests"] if "call" in x}

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

                eval_result = {
                    "name": workspace_dir_name,
                    "sum": total_runtime,
                    "passed": passed_ratio,
                    "num_passed": num_passed,
                    "num_tests": len(test_ids),
                }
            except json.JSONDecodeError:
                logger.error("Failed to parse test report JSON")
                # eval_result already has default values, no need to reassign

        # Save workspace as zip (if supported by workspace implementation)
        zip_dest = os.path.join(
            self.metadata.eval_output_dir, "repos", repo_name, f"{repo_name}.zip"
        )
        os.makedirs(os.path.dirname(zip_dest), exist_ok=True)

        # Try to copy workspace directory if the method is available
        try:
            if hasattr(workspace, "download_directory"):
                temp_zip = workspace.download_directory(repo_path)  # type: ignore[attr-defined]
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
            test_result=test_result,
            instruction=instruction,
            error=None,
            history=history,
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
        critic_name=args.critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
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
