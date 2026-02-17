#!/usr/bin/env python3
"""
SWE-Bench Evaluation Script

This script converts OpenHands output.jsonl format to SWE-Bench prediction format
and runs the SWE-Bench evaluation.

Usage:
    uv run swebench-eval <path_to_output.jsonl>
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.swebench import constants
from benchmarks.swebench.config import EVAL_DEFAULTS
from benchmarks.utils.constants import MODEL_NAME_OR_PATH
from benchmarks.utils.laminar import LaminarService
from benchmarks.utils.patch_utils import remove_files_from_patch
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


def convert_to_swebench_format(input_file: str, output_file: str) -> None:
    """
    Convert OpenHands output.jsonl to SWE-Bench prediction format.

    OpenHands format:
    {
        "instance_id": "django__django-11333",
        "test_result": {
            "git_patch": "diff --git a/file.py b/file.py\n..."
        },
        "instruction": "...",
        "error": null,
        "history": [...]
    }

    SWE-Bench format:
    {
        "instance_id": "django__django-11333",
        "model_patch": "diff --git a/file.py b/file.py\n...",
        "model_name_or_path": "<MODEL_NAME_OR_PATH>"
    }
    """
    logger.info(f"Converting {input_file} to SWE-Bench format: {output_file}")

    converted_count = 0
    error_count = 0

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                # Extract required fields
                instance_id = data.get("instance_id")
                if not instance_id:
                    logger.warning(f"Line {line_num}: Missing instance_id")
                    error_count += 1
                    continue

                # Extract git_patch from test_result
                test_result = data.get("test_result", {})
                git_patch = test_result.get("git_patch", "")

                if not git_patch:
                    logger.warning(
                        f"Line {line_num}: Missing or empty git_patch for {instance_id}"
                    )
                    # Still create entry with empty patch
                    git_patch = ""

                # postprocess git_patch
                git_patch = remove_files_from_patch(
                    git_patch, constants.SETUP_FILES_TO_REMOVE
                )

                # Create SWE-Bench format entry
                swebench_entry = {
                    "instance_id": instance_id,
                    "model_patch": git_patch,
                    "model_name_or_path": MODEL_NAME_OR_PATH,
                }

                # Write to output file
                outfile.write(json.dumps(swebench_entry) + "\n")
                converted_count += 1

            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON - {e}")
                error_count += 1
            except Exception as e:
                logger.error(f"Line {line_num}: Unexpected error - {e}")
                error_count += 1

    logger.info(
        f"Conversion complete: {converted_count} entries converted, "
        f"{error_count} errors"
    )

    if converted_count == 0:
        raise ValueError("No valid entries were converted")


def run_swebench_evaluation(
    predictions_file: str,
    run_id: str,
    dataset: str,
    workers: int,
    split: str,
    modal: bool,
    timeout: int,
) -> None:
    """
    Run SWE-Bench evaluation on the predictions file.

    Args:
        predictions_file: Path to the SWE-Bench format predictions file
        run_id: Unique identifier for this evaluation run
        dataset: SWE-Bench dataset to evaluate against
        workers: Number of workers to use for evaluation
        split: Dataset split to evaluate (e.g., 'test', 'dev')
        modal: Whether to use Modal for evaluation
        timeout: Timeout in seconds for evaluation
    """
    logger.info(f"Running SWE-Bench evaluation on {predictions_file}")

    try:
        # Get the directory of the predictions file
        predictions_path = Path(predictions_file)
        predictions_dir = predictions_path.parent
        predictions_filename = predictions_path.name

        # Try uv first, fall back to current Python interpreter
        try:
            uv_check = subprocess.run(
                ["uv", "--version"],
                capture_output=True,
                text=True,
            )
            uv_available = uv_check.returncode == 0
        except FileNotFoundError:
            uv_available = False

        if uv_available:
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "swebench.harness.run_evaluation",
            ]
        else:
            logger.info("uv not available, using current Python interpreter")
            cmd = [
                sys.executable,
                "-m",
                "swebench.harness.run_evaluation",
            ]

        cmd.extend([
            "--dataset_name",
            dataset,
            "--predictions_path",
            predictions_filename,
            "--max_workers",
            str(workers),
            "--run_id",
            run_id,
        ])

        # Add parameters
        cmd.extend(["--split", split])
        if modal:
            cmd.extend(["--modal", "true"])
        cmd.extend(["--timeout", str(timeout)])

        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Working directory: {predictions_dir}")
        logger.info("SWE-Bench evaluation output:")
        print("-" * 80)

        # Stream output directly to console, running from predictions file directory
        result = subprocess.run(cmd, text=True, cwd=predictions_dir)

        print("-" * 80)
        if result.returncode == 0:
            logger.info("SWE-Bench evaluation completed successfully")
        else:
            logger.error(
                f"SWE-Bench evaluation failed with return code {result.returncode}"
            )
            raise subprocess.CalledProcessError(result.returncode, cmd)

    except FileNotFoundError:
        logger.error(
            "SWE-Bench evaluation command not found. "
            "Make sure SWE-Bench is properly installed."
        )
        raise
    except Exception as e:
        logger.error(f"Error running SWE-Bench evaluation: {e}")
        raise


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert OpenHands output to SWE-Bench format and run evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run swebench-eval output.jsonl
    uv run swebench-eval /path/to/output.jsonl --dataset princeton-nlp/SWE-bench_Lite
    uv run swebench-eval output.jsonl --split test --run-id my_eval --modal --timeout 1800
        """,
    )

    parser.add_argument("input_file", help="Path to the OpenHands output.jsonl file")

    parser.add_argument(
        "--dataset",
        help="SWE-Bench dataset to evaluate against",
    )

    parser.add_argument(
        "--output-file",
        help="Output file for SWE-Bench format "
        "(default: input_file with .swebench.jsonl extension)",
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only convert format, skip running evaluation",
    )

    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to use when evaluating",
    )

    parser.add_argument(
        "--split",
        help="Dataset split to evaluate (e.g., 'test', 'dev')",
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Unique identifier for this evaluation run",
    )

    parser.add_argument(
        "--modal",
        dest="modal",
        action="store_true",
        help="Use Modal for evaluation",
    )

    parser.add_argument(
        "--no-modal",
        dest="modal",
        action="store_false",
        help="Do not use Modal for evaluation",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds for evaluation",
    )

    # Apply EVAL_DEFAULTS from config (for dataset, split, workers, modal, timeout)
    parser.set_defaults(**EVAL_DEFAULTS)

    args = parser.parse_args()

    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        sys.exit(1)

    if not input_file.suffix == ".jsonl":
        logger.warning(f"Input file does not have .jsonl extension: {input_file}")

    # Determine output file
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = input_file.with_suffix(".swebench.jsonl")

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Dataset: {args.dataset}")

    dest_report_path: Path | None = None

    try:
        # Convert format
        convert_to_swebench_format(str(input_file), str(output_file))

        if not args.skip_evaluation:
            # Run evaluation
            run_swebench_evaluation(
                str(output_file),
                args.run_id,
                args.dataset,
                args.workers,
                split=args.split,
                modal=args.modal,
                timeout=args.timeout,
            )

            # Move report file to input file directory with .report.json extension
            # SWE-Bench creates: {MODEL_NAME_OR_PATH}.{run_id}.json
            report_filename = f"{MODEL_NAME_OR_PATH}.{args.run_id}.json"
            report_path = output_file.parent / report_filename
            dest_report_path = input_file.with_suffix(".report.json")

            shutil.move(str(report_path), str(dest_report_path))
            logger.info(f"Moved report file to: {dest_report_path}")

            # Add benchmark field to the report
            with open(dest_report_path, "r") as f:
                report_data = json.load(f)
            if isinstance(args.dataset, str) and "/" in args.dataset:
                report_data["benchmark"] = args.dataset.split("/")[-1].lower()
            else:
                report_data["benchmark"] = str(args.dataset).lower()
            with open(dest_report_path, "w") as f:
                json.dump(report_data, f, indent=4)

            # Update Laminar datapoints with evaluation scores
            LaminarService.get().update_evaluation_scores(
                str(input_file), str(dest_report_path)
            )

        # Generate cost report as final step
        generate_cost_report(str(input_file))

        logger.info("Script completed successfully!")
        # Emit machine-readable report location for callers
        if not args.skip_evaluation and dest_report_path is not None:
            print(json.dumps({"report_json": str(dest_report_path)}))
        else:
            print(json.dumps({"report_json": ""}))

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
