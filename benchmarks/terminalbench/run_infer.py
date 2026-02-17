"""Terminal-Bench inference script using Harbor with openhands-sdk agent.

This script runs Terminal-Bench evaluation using Harbor as the harness
and openhands-sdk as the agent. Results are saved in a format compatible
with the standard evaluation pipeline.

Usage:
    uv run terminalbench-infer <llm_config_path> --dataset terminal-bench@head
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.terminalbench.config import HARBOR_DEFAULTS, INFER_DEFAULTS
from benchmarks.utils.evaluation_utils import construct_eval_output_dir
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import LLM, get_logger


logger = get_logger(__name__)

# Output filename for results
OUTPUT_FILENAME = "output.jsonl"


def check_harbor_installed() -> bool:
    """Check if harbor CLI is installed and available."""
    try:
        result = subprocess.run(
            ["harbor", "--version"],
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
    timeout: int = 3600,
) -> Path:
    """Run harbor evaluation with openhands-sdk agent.

    Args:
        llm: LLM configuration for the agent.
        dataset: Harbor dataset name (e.g., terminal-bench@head).
        output_dir: Directory to store output files.
        num_workers: Number of parallel workers.
        task_ids: Optional list of specific task IDs to run.
        timeout: Timeout per task in seconds.

    Returns:
        Path to the harbor output directory.
    """
    harbor_output_dir = Path(output_dir) / "harbor_output"
    harbor_output_dir.mkdir(parents=True, exist_ok=True)

    # Build harbor command
    cmd = [
        "harbor",
        "run",
        "-d",
        dataset,
        "-a",
        HARBOR_DEFAULTS["agent_name"],
        "-m",
        llm.model,
        "--output-dir",
        str(harbor_output_dir),
        "--max-workers",
        str(num_workers),
        "--timeout",
        str(timeout),
    ]

    # Add specific task IDs if provided
    if task_ids:
        for task_id in task_ids:
            cmd.extend(["--task-id", task_id])

    # Set up environment with LLM credentials
    env = os.environ.copy()
    if llm.api_key:
        # api_key can be str or SecretStr
        from pydantic import SecretStr

        if isinstance(llm.api_key, SecretStr):
            env["LLM_API_KEY"] = llm.api_key.get_secret_value()
        else:
            env["LLM_API_KEY"] = llm.api_key
    if llm.base_url:
        env["LLM_BASE_URL"] = llm.base_url
    env["LLM_MODEL"] = llm.model

    logger.info(f"Running harbor command: {' '.join(cmd)}")
    logger.info(f"Output directory: {harbor_output_dir}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=str(harbor_output_dir),
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
            "Harbor CLI not found. Please install harbor: pip install harbor-bench"
        )

    return harbor_output_dir


def convert_harbor_to_eval_output(
    harbor_output_dir: Path,
    eval_output_path: Path,
) -> None:
    """Convert harbor output (ATIF trajectories) to evaluation output format.

    Harbor stores results as ATIF trajectory JSON files. This function converts
    them to the standard JSONL format used by the evaluation pipeline.

    Args:
        harbor_output_dir: Path to harbor output directory.
        eval_output_path: Path to write the converted output.jsonl.
    """
    logger.info(f"Converting harbor output from {harbor_output_dir}")

    results: list[dict] = []

    # Look for trajectory files in harbor output
    # Harbor typically stores trajectories in trials/*/trajectory.json
    trajectory_patterns = [
        harbor_output_dir / "trials" / "*" / "trajectory.json",
        harbor_output_dir / "*" / "trajectory.json",
        harbor_output_dir / "trajectory.json",
    ]

    trajectory_files: list[Path] = []
    for pattern in trajectory_patterns:
        trajectory_files.extend(Path(pattern.parent).glob(pattern.name))

    # Also check for results.json which summarizes all trials
    results_file = harbor_output_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            harbor_results = json.load(f)
            logger.info(f"Found harbor results file with {len(harbor_results)} entries")

    if not trajectory_files and not results_file.exists():
        logger.warning(
            f"No trajectory files found in {harbor_output_dir}. "
            "Looking for any JSON files..."
        )
        trajectory_files = list(harbor_output_dir.rglob("*.json"))

    for traj_file in trajectory_files:
        if traj_file.name in ["results.json", "metadata.json"]:
            continue

        try:
            with open(traj_file) as f:
                trajectory = json.load(f)

            # Extract instance ID from trajectory or file path
            instance_id = trajectory.get("session_id") or traj_file.parent.name

            # Extract metrics from ATIF trajectory
            final_metrics = trajectory.get("final_metrics", {})
            steps = trajectory.get("steps", [])

            # Build history from steps
            history = []
            for step in steps:
                history.append(
                    {
                        "step_id": step.get("step_id"),
                        "source": step.get("source"),
                        "message": step.get("message"),
                        "timestamp": step.get("timestamp"),
                    }
                )

            # Create eval output entry
            eval_entry = {
                "instance_id": instance_id,
                "test_result": {
                    "trajectory_path": str(traj_file),
                    "total_steps": len(steps),
                    "final_metrics": final_metrics,
                },
                "instruction": (
                    steps[0].get("message", "") if steps else ""
                ),  # First user message
                "error": None,
                "history": history,
                "metrics": {
                    "total_prompt_tokens": final_metrics.get("total_prompt_tokens", 0),
                    "total_completion_tokens": final_metrics.get(
                        "total_completion_tokens", 0
                    ),
                    "total_cost_usd": final_metrics.get("total_cost_usd", 0.0),
                },
            }
            results.append(eval_entry)
            logger.info(f"Processed trajectory for instance: {instance_id}")

        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to process trajectory file {traj_file}: {e}")

    # Write results to output.jsonl
    with open(eval_output_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Wrote {len(results)} entries to {eval_output_path}")


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
        help="Harbor dataset name (e.g., terminal-bench@head, terminal-bench@2.0)",
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
        "--timeout",
        type=int,
        default=HARBOR_DEFAULTS["timeout"],
        help="Timeout per task in seconds",
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
            "  pip install harbor-bench\n"
            "  # or\n"
            "  uv pip install harbor-bench"
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
        "timeout": args.timeout,
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
                timeout=args.timeout,
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
