"""SkillsBench inference script using Harbor with openhands-sdk agent.

This script runs SkillsBench evaluation using Harbor as the harness
and openhands-sdk as the agent. Results are saved in a format compatible
with the standard evaluation pipeline.

Usage:
    uv run skillsbench-infer <llm_config_path> --dataset benchflow/skillsbench
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import SecretStr

from benchmarks.skillsbench.config import HARBOR_DEFAULTS, INFER_DEFAULTS
from benchmarks.utils.evaluation_utils import construct_eval_output_dir
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import LLM, get_logger


logger = get_logger(__name__)

# Output filename for results
OUTPUT_FILENAME = "output.jsonl"

SKILLSBENCH_REPO_URL = "https://github.com/benchflow-ai/skillsbench.git"
SKILLSBENCH_REPO_BRANCH = "main"
DATASET_CACHE_DIR = Path(__file__).parent / "data"
TASKS_CACHE_DIR = DATASET_CACHE_DIR / "tasks"
TASKS_METADATA_PATH = DATASET_CACHE_DIR / "source.json"
REGISTRY_DATASET_PREFIX = "benchflow/skillsbench"
INSTANCE_ID_PREFIX = "benchflow"


def check_harbor_installed() -> bool:
    """Check if harbor CLI is installed and available."""
    harbor_exe = HARBOR_DEFAULTS["harbor_executable"]
    try:
        result = subprocess.run(
            [harbor_exe, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_command(cmd: list[str], error_message: str) -> str:
    """Run a subprocess command and return stdout."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{error_message}: {stderr}")
    return result.stdout.strip()


def _get_supported_task_filter_flag(harbor_exe: str) -> str:
    """Detect whether Harbor expects --task-name or --include-task-name."""
    try:
        result = subprocess.run(
            [harbor_exe, "run", "--help"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "--include-task-name"

    help_text = f"{result.stdout}\n{result.stderr}"
    supported_flags = set(re.findall(r"(?<![\w-])--[a-z0-9-]+", help_text))
    if "--include-task-name" in supported_flags:
        return "--include-task-name"
    if "--task-name" in supported_flags:
        return "--task-name"
    return "--include-task-name"


def _get_supported_agent_name(harbor_exe: str) -> str:
    """Detect whether Harbor exposes the OpenHands agent as openhands or openhands-sdk."""
    try:
        result = subprocess.run(
            [harbor_exe, "run", "--help"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return HARBOR_DEFAULTS["agent_name"]

    help_text = f"{result.stdout}\n{result.stderr}"
    compact_help_text = re.sub(r"[^a-z0-9-]+", "", help_text.lower())
    if "openhands-sdk" in compact_help_text:
        return "openhands-sdk"
    if "openhands" in compact_help_text:
        return "openhands"
    return HARBOR_DEFAULTS["agent_name"]


def get_skillsbench_main_commit(
    repo_url: str = SKILLSBENCH_REPO_URL,
    branch: str = SKILLSBENCH_REPO_BRANCH,
) -> str:
    """Resolve the latest commit hash for the upstream SkillsBench branch."""
    stdout = _run_command(
        ["git", "ls-remote", repo_url, f"refs/heads/{branch}"],
        "Failed to resolve SkillsBench upstream commit",
    )
    commit_hash, _, ref = stdout.partition("\t")
    if not commit_hash or ref != f"refs/heads/{branch}":
        raise RuntimeError(
            f"Unexpected git ls-remote output for {repo_url} {branch}: {stdout}"
        )
    return commit_hash


def _load_cached_commit(metadata_path: Path = TASKS_METADATA_PATH) -> str | None:
    """Load the cached upstream commit hash for the local task snapshot."""
    if not metadata_path.is_file():
        return None

    try:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "Ignoring unreadable SkillsBench dataset metadata at %s: %s",
            metadata_path,
            e,
        )
        return None

    commit_hash = metadata.get("commit_hash")
    return commit_hash if isinstance(commit_hash, str) and commit_hash else None


def download_skillsbench_tasks(
    commit_hash: str,
    tasks_dir: Path = TASKS_CACHE_DIR,
    metadata_path: Path = TASKS_METADATA_PATH,
    repo_url: str = SKILLSBENCH_REPO_URL,
    branch: str = SKILLSBENCH_REPO_BRANCH,
) -> None:
    """Download only the SkillsBench tasks directory for a specific commit."""
    data_dir = tasks_dir.parent
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading SkillsBench tasks from %s@%s into %s",
        repo_url,
        commit_hash,
        tasks_dir,
    )

    with tempfile.TemporaryDirectory(dir=data_dir) as temp_dir:
        clone_dir = Path(temp_dir) / "skillsbench"
        _run_command(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                "--filter=blob:none",
                "--sparse",
                repo_url,
                str(clone_dir),
            ],
            "Failed to clone SkillsBench repository",
        )
        _run_command(
            ["git", "-C", str(clone_dir), "sparse-checkout", "set", "tasks"],
            "Failed to sparsely checkout SkillsBench tasks",
        )
        checked_out_commit = _run_command(
            ["git", "-C", str(clone_dir), "rev-parse", "HEAD"],
            "Failed to read cloned SkillsBench commit",
        )
        if checked_out_commit != commit_hash:
            raise RuntimeError(
                "Cloned SkillsBench commit does not match upstream HEAD: "
                f"expected {commit_hash}, got {checked_out_commit}"
            )

        source_tasks_dir = clone_dir / "tasks"
        if not source_tasks_dir.is_dir():
            raise RuntimeError(
                f"SkillsBench clone at {clone_dir} does not contain a tasks/ directory"
            )

        if tasks_dir.exists():
            shutil.rmtree(tasks_dir)
        shutil.copytree(source_tasks_dir, tasks_dir)

    metadata = {
        "repo_url": repo_url,
        "branch": branch,
        "commit_hash": commit_hash,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def ensure_skillsbench_tasks(
    tasks_dir: Path = TASKS_CACHE_DIR,
    metadata_path: Path = TASKS_METADATA_PATH,
    repo_url: str = SKILLSBENCH_REPO_URL,
    branch: str = SKILLSBENCH_REPO_BRANCH,
) -> Path:
    """Ensure a local SkillsBench task snapshot exists and matches upstream HEAD."""
    cached_commit = _load_cached_commit(metadata_path)
    has_cached_tasks = tasks_dir.is_dir() and any(tasks_dir.iterdir())

    try:
        upstream_commit = get_skillsbench_main_commit(repo_url=repo_url, branch=branch)
    except RuntimeError as e:
        if has_cached_tasks and cached_commit:
            logger.warning(
                "Failed to check SkillsBench upstream HEAD; using cached tasks from "
                "%s (%s): %s",
                tasks_dir,
                cached_commit,
                e,
            )
            return tasks_dir
        raise

    if has_cached_tasks and cached_commit == upstream_commit:
        logger.info(
            "Using cached SkillsBench tasks at %s (commit %s)",
            tasks_dir,
            upstream_commit,
        )
        return tasks_dir

    if has_cached_tasks:
        logger.info(
            "Refreshing SkillsBench tasks in %s from commit %s to %s",
            tasks_dir,
            cached_commit or "<unknown>",
            upstream_commit,
        )
    else:
        logger.info("No cached SkillsBench tasks found at %s; downloading", tasks_dir)

    download_skillsbench_tasks(
        commit_hash=upstream_commit,
        tasks_dir=tasks_dir,
        metadata_path=metadata_path,
        repo_url=repo_url,
        branch=branch,
    )
    return tasks_dir


def resolve_skillsbench_dataset(dataset: str) -> tuple[str, bool]:
    """Resolve the dataset argument to a synced local SkillsBench snapshot.

    Harbor 0.5.x validates ``--dataset`` values against the registry before
    starting a job. SkillsBench is not yet published in the public registry, so
    ``benchflow/skillsbench`` and versioned aliases like
    ``benchflow/skillsbench@1.0`` must be resolved to the locally synced Harbor
    task dataset generated by the SkillsBench adapter.
    """
    if dataset == REGISTRY_DATASET_PREFIX or dataset.startswith(
        f"{REGISTRY_DATASET_PREFIX}@"
    ):
        local_tasks_dir = ensure_skillsbench_tasks()
        return str(local_tasks_dir.resolve()), True
    raise ValueError(
        "Unsupported SkillsBench dataset source. Use the default synced "
        "SkillsBench snapshot or a SkillsBench dataset alias matching "
        "'benchflow/skillsbench@<version>'."
    )


def _normalize_task_filter_value(task_id: str, *, dataset_is_path: bool) -> str:
    """Normalize task filter values for Harbor's local-path dataset handling."""
    if dataset_is_path:
        return task_id.rsplit("/", 1)[-1]
    return task_id


def _canonicalize_instance_id(task_name: str) -> str:
    """Normalize SkillsBench task names to stable benchflow/<task-name> ids."""
    if "/" in task_name:
        return task_name
    return f"{INSTANCE_ID_PREFIX}/{task_name}"


def run_harbor_evaluation(
    llm: LLM,
    dataset: str,
    *,
    dataset_is_path: bool,
    output_dir: str,
    num_workers: int = 1,
    task_ids: list[str] | None = None,
    n_limit: int | None = None,
) -> Path:
    """Run harbor evaluation with openhands-sdk agent.

    Args:
        llm: LLM configuration for the agent.
        dataset: Synced SkillsBench task snapshot path or Harbor registry id.
        dataset_is_path: Whether ``dataset`` should be passed via ``--path``.
        output_dir: Directory to store output files.
        num_workers: Number of parallel workers.
        task_ids: Optional list of specific task IDs to run.
        n_limit: Optional maximum number of dataset tasks to run.

    Returns:
        Path to the harbor output directory.
    """
    harbor_output_dir = Path(output_dir) / "harbor_output"
    harbor_output_dir.mkdir(parents=True, exist_ok=True)
    harbor_exe = HARBOR_DEFAULTS["harbor_executable"]
    agent_name = _get_supported_agent_name(harbor_exe)
    task_filter_flag = _get_supported_task_filter_flag(harbor_exe)

    # Build harbor command using harbor CLI flags.
    # Use absolute path for --jobs-dir to avoid CWD-relative path issues.
    cmd = [
        harbor_exe,
        "run",
        "--path" if dataset_is_path else "-d",
        dataset,
        "-a",
        agent_name,
        "-m",
        llm.model,
        "--jobs-dir",
        str(harbor_output_dir.resolve()),
        "--n-concurrent",
        str(num_workers),
    ]

    # Add specific task names if provided
    if task_ids:
        for task_id in task_ids:
            cmd.extend(
                [
                    task_filter_flag,
                    _normalize_task_filter_value(
                        task_id, dataset_is_path=dataset_is_path
                    ),
                ]
            )

    if n_limit is not None:
        cmd.extend(["--n-tasks", str(n_limit)])

    logger.info(f"Running harbor command: {' '.join(cmd)}")
    logger.info(f"Output directory: {harbor_output_dir}")

    # harbor's openhands-sdk agent reads LLM credentials from the host process
    # environment (os.environ), not from --ae flags which go to the sandbox.
    env = os.environ.copy()
    if llm.api_key:
        api_key = (
            llm.api_key.get_secret_value()
            if isinstance(llm.api_key, SecretStr)
            else llm.api_key
        )
        env["LLM_API_KEY"] = api_key
    if llm.base_url:
        env["LLM_BASE_URL"] = llm.base_url

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            if (
                task_ids
                and task_filter_flag == "--task-name"
                and "No such option: --task-name" in result.stderr
            ):
                fallback_cmd = [
                    "--include-task-name" if part == "--task-name" else part
                    for part in cmd
                ]
                logger.warning(
                    "Harbor does not support --task-name; retrying with "
                    "--include-task-name"
                )
                result = subprocess.run(
                    fallback_cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                )

            if result.returncode != 0:
                logger.error(f"Harbor command failed with code {result.returncode}")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                raise RuntimeError(f"Harbor evaluation failed: {result.stderr}")

        logger.info("Harbor evaluation completed successfully")
        logger.info(f"stdout: {result.stdout}")

    except FileNotFoundError:
        raise RuntimeError(
            "Harbor CLI not found. Please install harbor: pip install harbor"
        )

    return harbor_output_dir


def _find_job_dir(harbor_output_dir: Path) -> Path:
    """Find the harbor job directory (timestamp-named) inside the output dir."""
    # Harbor creates a timestamp-named subdirectory (e.g., 2026-03-07__16-08-47)
    # containing result.json and trial subdirectories
    candidates = [
        d
        for d in harbor_output_dir.iterdir()
        if d.is_dir() and (d / "result.json").exists()
    ]
    if not candidates:
        raise RuntimeError(
            f"No harbor job directory found in {harbor_output_dir}. "
            f"Expected a timestamp-named directory containing result.json."
        )
    # Use the most recent job directory if multiple exist
    return sorted(candidates)[-1]


def convert_harbor_to_eval_output(
    harbor_output_dir: Path,
    eval_output_path: Path,
) -> None:
    """Convert harbor output to evaluation output format.

    Harbor stores trial results in a job directory structured as:
        harbor_output/TIMESTAMP/TRIAL_NAME/result.json

    Each trial's result.json contains task_name, verifier_result, agent_result,
    timing info, and exception details.

    Args:
        harbor_output_dir: Path to harbor output directory.
        eval_output_path: Path to write the converted output.jsonl.
    """
    logger.info(f"Converting harbor output from {harbor_output_dir}")

    job_dir = _find_job_dir(harbor_output_dir)
    logger.info(f"Using harbor job directory: {job_dir}")

    # Find trial result files (each trial dir has a result.json)
    result_files = list(job_dir.glob("*/result.json"))
    # Exclude the job-level result.json
    result_files = [f for f in result_files if f.parent != job_dir]

    if not result_files:
        raise RuntimeError(
            f"No trial result files found in {job_dir}. "
            f"Expected result.json files in trial subdirectories."
        )

    logger.info(f"Found {len(result_files)} trial results in {job_dir}")

    results: list[dict] = []
    errors: list[dict] = []

    for result_file in result_files:
        try:
            with open(result_file) as f:
                trial = json.load(f)

            instance_id = _canonicalize_instance_id(
                trial.get("task_name", result_file.parent.name)
            )

            # Check for exceptions
            if trial.get("exception_info"):
                errors.append(
                    {
                        "instance_id": instance_id,
                        "error": str(trial["exception_info"]),
                        "test_result": {},
                    }
                )
                continue

            # Extract verifier results
            verifier_result = trial.get("verifier_result", {})
            rewards = verifier_result.get("rewards", {})
            passed = rewards.get("reward", 0.0) > 0

            # Extract agent metrics
            agent_result = trial.get("agent_result", {})

            eval_entry = {
                "instance_id": instance_id,
                "test_result": {
                    "trial_name": trial.get("trial_name"),
                    "trial_uri": trial.get("trial_uri"),
                    "rewards": rewards,
                    "passed": passed,
                },
                "instruction": "",
                "error": None,
                "history": [],
                "metrics": {
                    "total_prompt_tokens": agent_result.get("n_input_tokens") or 0,
                    "total_completion_tokens": (
                        agent_result.get("n_output_tokens") or 0
                    ),
                    "total_cost_usd": agent_result.get("cost_usd") or 0.0,
                },
            }
            results.append(eval_entry)
            logger.info(
                f"Processed trial {instance_id}: reward={rewards.get('reward', 'N/A')}"
            )

        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to process result file {result_file}: {e}")
            errors.append(
                {
                    "instance_id": _canonicalize_instance_id(result_file.parent.name),
                    "error": str(e),
                    "test_result": {},
                }
            )

    if not results and not errors:
        raise RuntimeError(f"No trials processed from {harbor_output_dir}")

    if not results:
        logger.warning(
            f"All {len(errors)} trials failed in {harbor_output_dir}; "
            "writing error entries for downstream reporting"
        )

    # Write results to output.jsonl
    with open(eval_output_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
        for entry in errors:
            f.write(json.dumps(entry) + "\n")

    logger.info(
        f"Wrote {len(results)} successful + {len(errors)} failed entries "
        f"to {eval_output_path}"
    )


def load_task_ids_from_file(filepath: str) -> list[str]:
    """Load task IDs from a text file (one per line)."""
    task_ids = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                task_ids.append(line)
    return task_ids


def main() -> None:
    """Main entry point for skillsbench inference."""
    parser = argparse.ArgumentParser(
        description="Run SkillsBench evaluation with openhands-sdk via Harbor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full skillsbench evaluation using a local tasks/ snapshot synced from
    # https://github.com/benchflow-ai/skillsbench main (adapter-generated
    # Harbor tasks stored under benchmarks/skillsbench/data/tasks)
    uv run skillsbench-infer .llm_config/claude.json

    # Run specific tasks
    uv run skillsbench-infer .llm_config/claude.json --select tasks.txt

    # Versioned SkillsBench aliases also resolve to the synced local dataset
    uv run skillsbench-infer .llm_config/claude.json --dataset benchflow/skillsbench@1.0
        """,
    )

    parser.add_argument(
        "llm_config_path",
        type=str,
        help="Path to JSON LLM configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=INFER_DEFAULTS["dataset"],
        help=(
            "SkillsBench dataset source. The default value syncs tasks/ from the "
            "benchflow-ai/skillsbench main branch. Versioned aliases like "
            "benchflow/skillsbench@1.0 also resolve to the same local Harbor "
            "dataset because SkillsBench is not published in the public Harbor "
            "registry yet."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=INFER_DEFAULTS["output_dir"],
        help="Base output directory for evaluation results",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=INFER_DEFAULTS["num_workers"],
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        help="Maximum number of dataset tasks to run after Harbor filtering",
    )
    parser.add_argument(
        "--select",
        type=str,
        help="Path to text file containing task IDs to run (one per line)",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        action="append",
        help="Specific task ID to run (can be specified multiple times)",
    )
    parser.add_argument(
        "--note",
        type=str,
        help="Optional note for the evaluation run",
    )
    parser.add_argument(
        "--skip-harbor",
        action="store_true",
        help="Skip running harbor and only convert existing results",
    )

    args = parser.parse_args()

    # Validate LLM config
    if not os.path.isfile(args.llm_config_path):
        logger.error(f"LLM config file does not exist: {args.llm_config_path}")
        sys.exit(1)

    with open(args.llm_config_path) as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)
    logger.info(f"Using LLM: {llm.model}")

    # Check harbor installation
    if not args.skip_harbor and not check_harbor_installed():
        logger.error(
            "Harbor CLI is not installed. Please install it:\n"
            "  pip install harbor\n"
            "  # or\n"
            "  uv pip install harbor"
        )
        sys.exit(1)

    resolved_dataset = args.dataset
    dataset_is_path = False
    dataset_commit_hash: str | None = None
    if not args.skip_harbor:
        try:
            resolved_dataset, dataset_is_path = resolve_skillsbench_dataset(
                args.dataset
            )
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)
        if dataset_is_path and args.dataset == INFER_DEFAULTS["dataset"]:
            dataset_commit_hash = _load_cached_commit()

    # Construct output directory
    dataset_description = args.dataset.replace("/", "__").replace("@", "-")
    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=100,  # Not directly used but required for path construction
        eval_note=args.note,
    )

    logger.info(f"Output directory: {structured_output_dir}")
    os.makedirs(structured_output_dir, exist_ok=True)

    # Save metadata
    metadata = {
        "llm": llm.model_dump_json(),
        "dataset": args.dataset,
        "resolved_dataset": resolved_dataset,
        "dataset_is_path": dataset_is_path,
        "dataset_commit_hash": dataset_commit_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "harbor_agent": HARBOR_DEFAULTS["agent_name"],
        "note": args.note,
    }
    metadata_path = Path(structured_output_dir) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Collect task IDs if specified
    task_ids: list[str] | None = None
    if args.select:
        loaded_ids = load_task_ids_from_file(args.select)
        task_ids = loaded_ids
        logger.info(f"Loaded {len(loaded_ids)} task IDs from {args.select}")
    elif args.task_id:
        task_ids = list(args.task_id)
        logger.info(f"Running {len(task_ids)} specified task IDs")

    output_path = Path(structured_output_dir) / OUTPUT_FILENAME

    if not args.skip_harbor:
        # Run harbor evaluation
        try:
            harbor_output_dir = run_harbor_evaluation(
                llm=llm,
                dataset=resolved_dataset,
                dataset_is_path=dataset_is_path,
                output_dir=structured_output_dir,
                num_workers=args.num_workers,
                task_ids=task_ids,
                n_limit=args.n_limit,
            )

            # Convert harbor output to standard format
            convert_harbor_to_eval_output(
                harbor_output_dir=harbor_output_dir,
                eval_output_path=output_path,
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            sys.exit(1)
    else:
        # Skip harbor, just convert existing results
        harbor_output_dir = Path(structured_output_dir) / "harbor_output"
        if harbor_output_dir.exists():
            convert_harbor_to_eval_output(
                harbor_output_dir=harbor_output_dir,
                eval_output_path=output_path,
            )
        else:
            logger.error(f"No harbor output found at {harbor_output_dir}")
            sys.exit(1)

    # Generate cost report
    if output_path.exists():
        generate_cost_report(str(output_path))

    logger.info("SkillsBench inference completed!")
    print(json.dumps({"output_json": str(output_path)}))


if __name__ == "__main__":
    main()
