"""Terminal-Bench inference script using Harbor with openhands-sdk agent.

This script runs Terminal-Bench evaluation using Harbor as the harness
and openhands-sdk as the agent. Results are saved in a format compatible
with the standard evaluation pipeline.

Usage:
    uv run terminalbench-infer <llm_config_path> --dataset terminal-bench@2.0
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import SecretStr

from benchmarks.terminalbench.config import HARBOR_DEFAULTS, INFER_DEFAULTS
from benchmarks.utils.evaluation_utils import construct_eval_output_dir
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import LLM, get_logger


logger = get_logger(__name__)

# Output filename for results
OUTPUT_FILENAME = "output.jsonl"


def check_harbor_installed() -> bool:
    """Check if harbor CLI is installed and available."""
    harbor_exe = HARBOR_DEFAULTS["harbor_executable"]
    try:
        result = subprocess.run(
            [harbor_exe, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_harbor_evaluation(
    llm: LLM,
    dataset: str,
    output_dir: str,
    num_workers: int = 1,
    task_ids: list[str] | None = None,
    n_limit: int | None = None,
) -> Path:
    """Run harbor evaluation with openhands-sdk agent.

    Args:
        llm: LLM configuration for the agent.
        dataset: Harbor dataset name (e.g., terminal-bench@2.0).
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

    # Build harbor command using harbor CLI flags.
    # Use absolute path for --jobs-dir to avoid CWD-relative path issues.
    cmd = [
        harbor_exe,
        "run",
        "-d",
        dataset,
        "-a",
        HARBOR_DEFAULTS["agent_name"],
        "-m",
        llm.model,
        "--jobs-dir",
        str(harbor_output_dir.resolve()),
        "--n-concurrent",
        str(num_workers),
    ]

    # Pass LLM credentials as agent environment variables
    if llm.api_key:
        api_key = (
            llm.api_key.get_secret_value()
            if isinstance(llm.api_key, SecretStr)
            else llm.api_key
        )
        cmd.extend(["--ae", f"LLM_API_KEY={api_key}"])
    if llm.base_url:
        cmd.extend(["--ae", f"LLM_BASE_URL={llm.base_url}"])

    # Add specific task names if provided
    if task_ids:
        for task_id in task_ids:
            cmd.extend(["--task-name", task_id])

    if n_limit is not None:
        cmd.extend(["--n-tasks", str(n_limit)])

    logger.info(f"Running harbor command: {' '.join(cmd)}")
    logger.info(f"Output directory: {harbor_output_dir}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
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

            instance_id = trial.get("task_name", result_file.parent.name)

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
                    "instance_id": result_file.parent.name,
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
    """Main entry point for terminal-bench inference."""
    parser = argparse.ArgumentParser(
        description="Run Terminal-Bench evaluation with openhands-sdk via Harbor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full terminal-bench evaluation
    uv run terminalbench-infer .llm_config/claude.json

    # Run specific tasks
    uv run terminalbench-infer .llm_config/claude.json --select tasks.txt

    # Run with custom dataset version
    uv run terminalbench-infer .llm_config/claude.json --dataset terminal-bench@2.0
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
        help="Harbor dataset name (e.g., terminal-bench@2.0)",
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
        task_ids = list(args.task_id)  # Convert to ensure it's a list
        logger.info(f"Running {len(task_ids)} specified task IDs")

    output_path = Path(structured_output_dir) / OUTPUT_FILENAME

    if not args.skip_harbor:
        # Run harbor evaluation
        try:
            harbor_output_dir = run_harbor_evaluation(
                llm=llm,
                dataset=args.dataset,
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

    logger.info("Terminal-Bench inference completed!")
    print(json.dumps({"output_json": str(output_path)}))


if __name__ == "__main__":
    main()
