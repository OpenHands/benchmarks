"""SkillsBench inference script using the benchflow SDK.

This script runs SkillsBench evaluation using `benchflow job` as the harness
and `openhands` as the default agent. Results are saved in a format compatible
with the standard evaluation pipeline.

Usage:
    uv run skillsbench-infer <llm_config_path>

    # Run specific tasks
    uv run skillsbench-infer <llm_config_path> --select tasks.txt
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

import yaml
from pydantic import SecretStr

from benchmarks.skillsbench.config import BENCHFLOW_DEFAULTS, INFER_DEFAULTS
from benchmarks.utils.evaluation_utils import construct_eval_output_dir
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import LLM, get_logger


logger = get_logger(__name__)

# Matches benchflow 0.3.0 job directory names: YYYY-MM-DD__HH-MM-SS
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}$")

# "Total cost: $0.0487"
_COST_RE = re.compile(r"Total cost:\s*\$([0-9]+(?:\.[0-9]+)?)")
# "Tokens: ↑ input 404.21K • ... • ↓ output 7.83K"
_TOKENS_RE = re.compile(r"↑ input\s+([\d.]+)([KMB]?)\b.*?↓ output\s+([\d.]+)([KMB]?)\b")

OUTPUT_FILENAME = "output.jsonl"

TASK_REPOS = {
    "skillsbench": {
        "repo": "https://github.com/benchflow-ai/skillsbench.git",
        "subdir": "tasks",
    }
}

_DIRECT_PROVIDER_ENV_VARS: dict[str, tuple[tuple[str, ...], str | None]] = {
    "anthropic": (("ANTHROPIC_API_KEY",), "ANTHROPIC_BASE_URL"),
    "gemini": (("GEMINI_API_KEY", "GOOGLE_API_KEY"), "GEMINI_BASE_URL"),
    "google": (("GEMINI_API_KEY", "GOOGLE_API_KEY"), "GEMINI_BASE_URL"),
    "openai": (("OPENAI_API_KEY",), "OPENAI_BASE_URL"),
}


def _infer_direct_provider(model: str) -> str | None:
    """Infer the provider prefix for direct model names.

    Examples:
      - gemini/gemini-2.5-pro -> gemini
      - anthropic/claude-sonnet-4-5 -> anthropic
      - litellm_proxy/anthropic/... -> None (proxy config uses LLM_* vars)
    """
    if not model or model.startswith("litellm_proxy/"):
        return None
    if "/" in model:
        provider = model.split("/", 1)[0].lower()
        if provider in _DIRECT_PROVIDER_ENV_VARS:
            return provider
    return None


def _build_benchflow_agent_env(llm: LLM) -> dict[str, str]:
    """Build the sandbox environment for benchflow's openhands agent.

    Only LLM-specific variables are returned — these go INTO the sandbox
    container via the ``agent_env`` YAML key.  The calling process inherits
    the host environment normally; dumping ``os.environ`` here would leak
    the entire host env into every container.
    """
    env: dict[str, str] = {}
    api_key: str | None = None
    if llm.api_key:
        api_key = (
            llm.api_key.get_secret_value()
            if isinstance(llm.api_key, SecretStr)
            else llm.api_key
        )
        env["LLM_API_KEY"] = api_key
    if llm.base_url:
        env["LLM_BASE_URL"] = llm.base_url

    provider = _infer_direct_provider(llm.model)
    if provider and api_key:
        key_vars, base_url_var = _DIRECT_PROVIDER_ENV_VARS[provider]
        for var_name in key_vars:
            env[var_name] = api_key
        if llm.base_url and base_url_var:
            env[base_url_var] = llm.base_url

    return env


def check_benchflow_installed() -> bool:
    """Check if benchflow CLI is installed and available.

    Tries ``bench`` first (current name), then falls back to the legacy
    ``benchflow`` binary.
    """
    for cmd in ("bench", "benchflow"):
        try:
            result = subprocess.run(
                [cmd, "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def _resolve_task_repo(dataset: str) -> tuple[str, dict[str, str]]:
    """Map a benchflow dataset name to its task repository metadata."""
    dataset_name = dataset.split("@", 1)[0].split("/")[-1]
    try:
        return dataset_name, TASK_REPOS[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported SkillsBench dataset: {dataset!r}. "
            f"Known datasets: {sorted(TASK_REPOS)}"
        ) from exc


def ensure_tasks(
    dataset: str,
    tasks_dir: Path,
    task_ids: list[str] | None = None,
) -> None:
    """Download tasks for a benchflow dataset into tasks_dir.

    BenchFlow 0.3.0 does not expose ``benchflow tasks pull``, so we clone the
    benchmark task repository directly when the local tasks directory is empty.

    When *task_ids* is provided, a sparse checkout is used so only the
    requested task subdirectories are downloaded — much faster than a full
    clone for large repos.
    """
    if tasks_dir.exists() and any(tasks_dir.iterdir()):
        logger.info(f"Tasks already present in {tasks_dir}, skipping download")
        return

    _, repo_info = _resolve_task_repo(dataset)
    tasks_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = tasks_dir.parent / "_clone"
    if clone_dir.exists():
        shutil.rmtree(clone_dir, ignore_errors=True)

    subdir = repo_info.get("subdir", "")

    if task_ids:
        # Sparse checkout: only download the specific task directories
        short_names = [tid.split("/")[-1] for tid in task_ids]

        cmd_clone = [
            "git",
            "clone",
            "--no-checkout",
            "--depth",
            "1",
            repo_info["repo"],
            str(clone_dir),
        ]
        logger.info(f"Sparse clone: {' '.join(cmd_clone)}")
        result = subprocess.run(cmd_clone, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"task download failed: {result.stderr}")

        # Init sparse-checkout and set the desired paths
        subprocess.run(
            ["git", "-C", str(clone_dir), "sparse-checkout", "init", "--cone"],
            capture_output=True,
            text=True,
            check=True,
        )
        sparse_paths = [f"{subdir}/{name}" if subdir else name for name in short_names]
        subprocess.run(
            ["git", "-C", str(clone_dir), "sparse-checkout", "set", *sparse_paths],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(clone_dir), "checkout"],
            capture_output=True,
            text=True,
            check=True,
        )
    else:
        # Full shallow clone
        cmd = ["git", "clone", "--depth", "1", repo_info["repo"], str(clone_dir)]
        logger.info(f"Downloading tasks: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to clone tasks: {result.stderr}")
            raise RuntimeError(f"task download failed: {result.stderr}")

    try:
        source_dir = clone_dir / subdir if subdir else clone_dir

        for entry in source_dir.iterdir():
            target = tasks_dir / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, target)
    finally:
        shutil.rmtree(clone_dir, ignore_errors=True)

    logger.info(f"Tasks downloaded to {tasks_dir}")


def run_benchflow_job(
    llm: LLM,
    tasks_dir: Path,
    jobs_dir: Path,
    num_workers: int = 1,
    task_ids: list[str] | None = None,
) -> Path:
    """Run benchflow job command.

    Args:
        llm: LLM configuration for the agent.
        tasks_dir: Path to directory containing task subdirectories.
        jobs_dir: Directory for benchflow job output.
        num_workers: Number of parallel workers (concurrency).
        task_ids: Optional list of task IDs to filter (short names, not full paths).

    Returns:
        Path to jobs_dir.
    """
    jobs_dir.mkdir(parents=True, exist_ok=True)

    agent_env = _build_benchflow_agent_env(llm)
    # Ubuntu 24.04 enforces PEP 668 and blocks bare `pip install` without
    # --break-system-packages. benchflow's openhands install_cmd uses plain
    # `pip install openhands`, which silently fails (exit 0) on Ubuntu 24.04,
    # causing "Agent openhands install failed (rc=1)". Setting this env var
    # makes pip skip the restriction without modifying the install_cmd.
    agent_env.setdefault("PIP_BREAK_SYSTEM_PACKAGES", "1")
    config = {
        "tasks_dir": str(tasks_dir),
        "jobs_dir": str(jobs_dir.resolve()),
        "agent": BENCHFLOW_DEFAULTS["agent_name"],
        "model": llm.model,
        "environment": "docker",
        "concurrency": num_workers,
        # OpenHands is installed inside the sandbox as root by benchflow's
        # registry install command. Running as the default "agent" user can
        # lose access to that binary on some task images.
        "sandbox_user": None,
        "agent_env": agent_env,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="benchflow-job-", delete=False
    ) as tmp:
        yaml.safe_dump(config, tmp, sort_keys=False)
        config_path = tmp.name

    # Prefer `bench eval create` (current), fall back to legacy `benchflow job`
    bench_bin = shutil.which("bench") or shutil.which("benchflow") or "bench"
    if "benchflow" in bench_bin:
        cmd = [bench_bin, "job", "--config", config_path]
    else:
        cmd = [bench_bin, "eval", "create", "-f", config_path]

    logger.info(f"Running: {' '.join(cmd)}")

    # Inject LLM vars into the host process env so benchflow's provider
    # resolution can pick them up; the subprocess inherits normally (env=None).
    host_env = os.environ.copy()
    host_env.update(agent_env)
    result = subprocess.run(cmd, capture_output=True, text=True, env=host_env)
    Path(config_path).unlink(missing_ok=True)

    if result.returncode != 0:
        logger.error(f"benchflow job failed (code {result.returncode})")
        logger.error(f"stdout: {result.stdout}")
        logger.error(f"stderr: {result.stderr}")
        raise RuntimeError(f"benchflow job failed: {result.stderr}")

    logger.info("benchflow job completed")
    logger.info(f"stdout: {result.stdout}")

    return jobs_dir


def _extract_trial_metrics(trial_dir: Path) -> dict:
    """Extract token/cost metrics from benchflow 0.3.0 trial output files.

    benchflow 0.3.0 does not write cost/token fields to result.json.
    Instead, metrics are read from:
      1. agent/trajectory.json → final_metrics (harbor-format agent)
      2. agent/openhands.txt  → "Total cost:" and "Tokens:" lines (ACP agent)
    """
    # 1. Harbor-format trajectory.json written by openhands-sdk agent
    traj_file = trial_dir / "agent" / "trajectory.json"
    if traj_file.exists():
        try:
            with open(traj_file) as f:
                traj = json.load(f)
            fm = traj.get("final_metrics") or {}
            if fm:
                return {
                    "total_prompt_tokens": int(fm.get("total_prompt_tokens") or 0),
                    "total_completion_tokens": int(
                        fm.get("total_completion_tokens") or 0
                    ),
                    "total_cost_usd": float(fm.get("total_cost_usd") or 0.0),
                }
        except (json.JSONDecodeError, OSError):
            pass

    # 2. ACP agent log written by openhands acp (benchflow 0.3.0 native)
    def _parse_token_count(value: str, suffix: str) -> int:
        n = float(value)
        return int(
            n * {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(suffix.upper(), 1)
        )

    for log_name in ("openhands.txt", "openhands_sdk.txt"):
        log_file = trial_dir / "agent" / log_name
        if not log_file.exists():
            continue
        try:
            text = log_file.read_text(errors="replace")
            cost_usd = 0.0
            prompt_tokens = 0
            completion_tokens = 0
            m = _COST_RE.search(text)
            if m:
                cost_usd = float(m.group(1))
            m = _TOKENS_RE.search(text)
            if m:
                prompt_tokens = _parse_token_count(m.group(1), m.group(2))
                completion_tokens = _parse_token_count(m.group(3), m.group(4))
            if cost_usd or prompt_tokens:
                return {
                    "total_prompt_tokens": prompt_tokens,
                    "total_completion_tokens": completion_tokens,
                    "total_cost_usd": cost_usd,
                }
        except OSError:
            pass

    return {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_cost_usd": 0.0,
    }


def convert_benchflow_to_eval_output(
    jobs_dir: Path,
    eval_output_path: Path,
    task_ids: list[str] | None = None,
) -> None:
    """Convert benchflow job output to standard evaluation output format.

    benchflow 0.3.0 stores trial results as:
        jobs_dir/YYYY-MM-DD__HH-MM-SS/TASK_NAME__UUID8/result.json

    Each result.json contains task_name, rewards, error, verifier_error, and timing.

    Args:
        jobs_dir: Path to benchflow jobs directory.
        eval_output_path: Path to write output.jsonl.
        task_ids: Optional filter for specific task IDs (short names).
    """
    logger.info(f"Converting benchflow output from {jobs_dir}")

    # benchflow 0.3.0 writes:
    #   jobs/summary.json
    #   jobs/TIMESTAMP/TRIAL_NAME/result.json
    # while older local outputs may place results directly under jobs/.
    job_dirs = [d for d in jobs_dir.iterdir() if d.is_dir()]
    timestamp_job_dirs = [d for d in job_dirs if _TIMESTAMP_RE.match(d.name)]

    if timestamp_job_dirs:
        selected_job_dir = sorted(timestamp_job_dirs)[-1]
        logger.info(f"Using benchflow job directory: {selected_job_dir}")
        task_dirs = [d for d in selected_job_dir.iterdir() if d.is_dir()]
    else:
        task_dirs = job_dirs

    if not task_dirs:
        raise RuntimeError(f"No task directories found in {jobs_dir}")

    if task_ids:
        short_ids = {tid.split("/")[-1] for tid in task_ids}
        task_dirs = [d for d in task_dirs if d.name.split("__")[0] in short_ids]

    logger.info(f"Processing {len(task_dirs)} task directories")

    results: list[dict] = []
    errors: list[dict] = []

    for task_dir in sorted(task_dirs):
        # Find the trial result — benchflow writes trial-0/result.json
        trial_results = list(task_dir.glob("trial-*/result.json"))
        if not trial_results:
            # Fall back to a direct result.json
            direct = task_dir / "result.json"
            if direct.exists():
                trial_results = [direct]

        if not trial_results:
            logger.warning(f"No result.json found in {task_dir}, skipping")
            errors.append(
                {
                    "instance_id": f"benchflow/{task_dir.name}",
                    "error": "No result.json found",
                    "test_result": {},
                }
            )
            continue

        # Use the last trial (highest retry index)
        result_file = sorted(trial_results)[-1]

        try:
            with open(result_file) as f:
                trial = json.load(f)

            task_basename = task_dir.name.split("__")[0]
            task_name = trial.get("task_name") or f"benchflow/{task_basename}"
            # Normalise to benchflow/<name> form
            if "/" not in task_name:
                task_name = f"benchflow/{task_name}"

            error = trial.get("error")
            verifier_error = trial.get("verifier_error")

            if error or verifier_error:
                errors.append(
                    {
                        "instance_id": task_name,
                        "error": str(error or verifier_error),
                        "test_result": {},
                    }
                )
                continue

            rewards = trial.get("rewards") or {}
            passed = bool(rewards.get("reward", 0.0))

            eval_entry = {
                "instance_id": task_name,
                "test_result": {
                    "rewards": rewards,
                    "passed": passed,
                },
                "instruction": "",
                "error": None,
                "history": [],
                "metrics": _extract_trial_metrics(result_file.parent),
            }
            results.append(eval_entry)
            logger.info(f"Processed {task_name}: reward={rewards.get('reward', 'N/A')}")

        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to read {result_file}: {e}")
            errors.append(
                {
                    "instance_id": f"benchflow/{task_dir.name}",
                    "error": str(e),
                    "test_result": {},
                }
            )

    if not results and not errors:
        raise RuntimeError(f"No trials processed from {jobs_dir}")

    if not results:
        logger.warning(
            f"All {len(errors)} trials failed; writing error entries for reporting"
        )

    with open(eval_output_path, "w") as f:
        for entry in results + errors:
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
        description="Run SkillsBench evaluation with benchflow and openhands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full skillsbench evaluation
    uv run skillsbench-infer .llm_config/claude.json

    # Run specific tasks from a file
    uv run skillsbench-infer .llm_config/claude.json --select tasks.txt

    # Run with more concurrency
    uv run skillsbench-infer .llm_config/claude.json --num-workers 4
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
        help="benchflow dataset name (e.g., benchflow/skillsbench)",
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
        help="Number of parallel workers (concurrency)",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        help="Maximum number of tasks to run",
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
        "--skip-run",
        action="store_true",
        help="Skip running benchflow and only convert existing results",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.llm_config_path):
        logger.error(f"LLM config file does not exist: {args.llm_config_path}")
        sys.exit(1)

    with open(args.llm_config_path) as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)
    logger.info(f"Using LLM: {llm.model}")

    if not args.skip_run and not check_benchflow_installed():
        logger.error(
            "benchflow CLI is not installed. Please install it:\n"
            "  uv tool install benchflow==0.3.0\n"
            "  # or\n"
            "  pip install benchflow==0.3.0\n"
            "  # or\n"
            "  uv pip install benchflow==0.3.0"
        )
        sys.exit(1)

    dataset_description = args.dataset.replace("/", "__").replace("@", "-")
    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=100,
        eval_note=args.note,
    )

    logger.info(f"Output directory: {structured_output_dir}")
    os.makedirs(structured_output_dir, exist_ok=True)

    metadata = {
        "llm": llm.model_dump_json(),
        "dataset": args.dataset,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchflow_agent": BENCHFLOW_DEFAULTS["agent_name"],
        "note": args.note,
    }
    metadata_path = Path(structured_output_dir) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    task_ids: list[str] | None = None
    if args.select:
        task_ids = load_task_ids_from_file(args.select)
        logger.info(f"Loaded {len(task_ids)} task IDs from {args.select}")
    elif args.task_id:
        task_ids = list(args.task_id)
        logger.info(f"Running {len(task_ids)} specified task IDs")

    tasks_dir = Path(structured_output_dir) / "tasks"
    jobs_dir = Path(structured_output_dir) / "jobs"
    output_path = Path(structured_output_dir) / OUTPUT_FILENAME

    if not args.skip_run:
        try:
            ensure_tasks(args.dataset, tasks_dir, task_ids=task_ids)

            # Apply n_limit by slicing available task directories
            effective_task_dirs = tasks_dir
            if args.n_limit is not None or task_ids is not None:
                all_dirs = sorted(d for d in tasks_dir.iterdir() if d.is_dir())
                if task_ids:
                    short_ids = {tid.split("/")[-1] for tid in task_ids}
                    all_dirs = [d for d in all_dirs if d.name in short_ids]
                if args.n_limit is not None:
                    all_dirs = all_dirs[: args.n_limit]

                # Write a filtered tasks dir symlink tree
                filtered_tasks_dir = Path(structured_output_dir) / "tasks_filtered"
                filtered_tasks_dir.mkdir(exist_ok=True)
                for d in all_dirs:
                    link = filtered_tasks_dir / d.name
                    if not link.exists():
                        link.symlink_to(d.resolve())
                effective_task_dirs = filtered_tasks_dir

            run_benchflow_job(
                llm=llm,
                tasks_dir=effective_task_dirs,
                jobs_dir=jobs_dir,
                num_workers=args.num_workers,
                task_ids=task_ids,
            )

            convert_benchflow_to_eval_output(
                jobs_dir=jobs_dir,
                eval_output_path=output_path,
                task_ids=task_ids,
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            sys.exit(1)
    else:
        if jobs_dir.exists():
            convert_benchflow_to_eval_output(
                jobs_dir=jobs_dir,
                eval_output_path=output_path,
                task_ids=task_ids,
            )
        else:
            logger.error(f"No jobs output found at {jobs_dir}")
            sys.exit(1)

    if output_path.exists():
        generate_cost_report(str(output_path))

    logger.info("SkillsBench inference completed!")
    print(json.dumps({"output_json": str(output_path)}))


if __name__ == "__main__":
    main()
