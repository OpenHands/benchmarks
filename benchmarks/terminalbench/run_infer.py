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

from benchmarks.terminalbench.config import HARBOR_DEFAULTS, INFER_DEFAULTS
from benchmarks.utils.evaluation_utils import construct_eval_output_dir
from benchmarks.utils.harbor import (
    HarborCredentialMode,
    check_harbor_installed as _check_harbor_installed,
    convert_harbor_to_eval_output,
    get_supported_task_filter_flag,
    run_harbor_evaluation as _run_harbor_evaluation,
)
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import LLM, get_logger


logger = get_logger(__name__)

# Output filename for results
OUTPUT_FILENAME = "output.jsonl"


def check_harbor_installed() -> bool:
    """Check if harbor CLI is installed and available."""
    return _check_harbor_installed(
        HARBOR_DEFAULTS["harbor_executable"],
        probe_arg="--version",
    )


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
    return _run_harbor_evaluation(
        llm=llm,
        dataset=dataset,
        output_dir=output_dir,
        harbor_executable=HARBOR_DEFAULTS["harbor_executable"],
        agent_name=HARBOR_DEFAULTS["agent_name"],
        num_workers=num_workers,
        task_ids=task_ids,
        n_limit=n_limit,
        task_filter_flag=get_supported_task_filter_flag(
            HARBOR_DEFAULTS["harbor_executable"]
        ),
        credential_mode=HarborCredentialMode.AGENT_ENV_FLAGS,
        subprocess_run=subprocess.run,
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
