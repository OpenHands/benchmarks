#!/usr/bin/env python3
"""
SWE-Smith Evaluation Script

This script converts OpenHands output.jsonl format to SWE-Smith prediction format
and runs the SWE-Smith evaluation.

Usage:
    uv run swesmith-eval <path_to_output.jsonl> --run-id <run_id> --dataset <path_to_dataset>
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from swesmith.harness.eval import main as swesmith_eval_main

import benchmarks.swesmith.profiles  # noqa: F401 — registers custom profiles
from benchmarks.swesmith import constants
from benchmarks.swesmith.config import EVAL_DEFAULTS
from benchmarks.utils.constants import MODEL_NAME_OR_PATH
from benchmarks.utils.laminar import LaminarService
from benchmarks.utils.patch_utils import remove_files_from_patch
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


def convert_to_swesmith_format(input_file: str, output_file: str) -> None:
    """
    Convert OpenHands output.jsonl to SWE-Smith prediction format.

    OpenHands format:
    {
        "instance_id": "repo__name.hash__ig_llm",
        "test_result": {
            "git_patch": "diff --git a/file.py b/file.py\n..."
        },
        ...
    }

    SWE-Smith format:
    {
        "instance_id": "repo__name.hash__ig_llm",
        "model_patch": "diff --git a/file.py b/file.py\n...",
        "model_name_or_path": "<MODEL_NAME_OR_PATH>"
    }
    """
    logger.info(f"Converting {input_file} to SWE-Smith format: {output_file}")

    converted_count = 0
    error_count = 0

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                instance_id = data.get("instance_id")
                if not instance_id:
                    logger.warning(f"Line {line_num}: Missing instance_id")
                    error_count += 1
                    continue

                test_result = data.get("test_result", {})
                git_patch = test_result.get("git_patch", "")

                if not git_patch:
                    logger.warning(
                        f"Line {line_num}: Missing or empty git_patch for {instance_id}"
                    )
                    git_patch = ""

                git_patch = remove_files_from_patch(
                    git_patch, constants.SETUP_FILES_TO_REMOVE
                )

                swesmith_entry = {
                    "instance_id": instance_id,
                    "model_patch": git_patch,
                    "model_name_or_path": MODEL_NAME_OR_PATH,
                }

                outfile.write(json.dumps(swesmith_entry) + "\n")
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


def run_swesmith_evaluation(
    predictions_file: str,
    run_id: str,
    dataset: str,
    workers: int = EVAL_DEFAULTS["workers"],
    f2p_only: bool = False,
    instance_ids: list[str] | None = None,
    report_only: bool = False,
    redo_existing: bool = False,
) -> None:
    """
    Run SWE-Smith evaluation on the predictions file.

    Calls swesmith.harness.eval directly as a Python API (not subprocess).
    Custom profiles from benchmarks.swesmith.profiles are auto-registered
    at import time, making them available to the swesmith harness.

    Args:
        predictions_file: Path to the SWE-Smith format predictions file
        run_id: Unique identifier for this evaluation run
        dataset: Path to SWE-Smith dataset file (.json or .jsonl)
        workers: Number of workers to use for evaluation
        f2p_only: Run evaluation using only files with fail-to-pass tests
        instance_ids: Instance IDs to evaluate (supports glob patterns)
        report_only: Regenerate reports only, skip running evaluations
        redo_existing: Redo already-completed evaluation instances
    """
    logger.info(f"Running SWE-Smith evaluation on {predictions_file}")

    predictions_path = Path(predictions_file)
    predictions_dir = predictions_path.parent

    # Resolve dataset to absolute path before changing cwd
    dataset_abs = str(Path(dataset).resolve())

    logger.info(f"Working directory: {predictions_dir}")

    # swesmith writes logs relative to cwd, so we temporarily change to
    # the predictions directory (same effect as subprocess cwd=).
    original_cwd = os.getcwd()
    os.chdir(predictions_dir)
    try:
        swesmith_eval_main(
            run_id=run_id,
            workers=workers,
            predictions_path=predictions_path.name,
            dataset_path=dataset_abs,
            f2p_only=f2p_only,
            instance_ids=instance_ids,
            report_only=report_only,
            redo_existing=redo_existing,
        )
        logger.info("SWE-Smith evaluation completed successfully")
    except Exception as e:
        logger.error(f"SWE-Smith evaluation failed: {e}")
        raise
    finally:
        os.chdir(original_cwd)


def main() -> None:
    """Main entry point for the script."""
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Convert OpenHands output to SWE-Smith format and run evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run swesmith-eval output.jsonl --run-id my_eval --dataset /path/to/dataset.json
    uv run swesmith-eval output.jsonl --run-id test --dataset /path/to/dataset.json --skip-evaluation
    uv run swesmith-eval output.jsonl --run-id fast --dataset /path/to/dataset.json --f2p-only
    uv run swesmith-eval output.jsonl --run-id filtered --dataset /path/to/dataset.json --instance-ids "repo__name.*"
        """,
    )

    parser.add_argument("input_file", help="Path to the OpenHands output.jsonl file")

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to SWE-Smith dataset file (.json or .jsonl)",
    )

    parser.add_argument(
        "--output-file",
        help="Output file for SWE-Smith format "
        "(default: input_file with .swesmith.jsonl extension)",
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only convert format, skip running evaluation",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=EVAL_DEFAULTS["workers"],
        help=f"Number of workers to use when evaluating (default: {EVAL_DEFAULTS['workers']})",
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Unique identifier for this evaluation run",
    )

    parser.add_argument(
        "--f2p-only",
        action="store_true",
        help="Run evaluation using only files with fail-to-pass tests (faster)",
    )

    parser.add_argument(
        "--instance-ids",
        nargs="+",
        help="Instance IDs to evaluate (supports glob patterns like 'repo__name.*')",
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate reports only, skip running evaluations",
    )

    parser.add_argument(
        "--redo-existing",
        action="store_true",
        help="Redo already-completed evaluation instances",
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        sys.exit(1)

    if not input_file.suffix == ".jsonl":
        logger.warning(f"Input file does not have .jsonl extension: {input_file}")

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = input_file.with_suffix(".swesmith.jsonl")

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Dataset: {args.dataset}")

    dest_report_path: Path | None = None

    try:
        convert_to_swesmith_format(str(input_file), str(output_file))

        if not args.skip_evaluation:
            run_swesmith_evaluation(
                str(output_file),
                args.run_id,
                args.dataset,
                args.workers,
                f2p_only=args.f2p_only,
                instance_ids=args.instance_ids,
                report_only=args.report_only,
                redo_existing=args.redo_existing,
            )

            # swesmith creates: logs/run_evaluation/{run_id}/report.json relative to cwd
            report_path = (
                output_file.parent
                / "logs"
                / "run_evaluation"
                / args.run_id
                / "report.json"
            )
            dest_report_path = input_file.with_suffix(".report.json")

            shutil.move(str(report_path), str(dest_report_path))
            logger.info(f"Moved report file to: {dest_report_path}")

            LaminarService.get().update_evaluation_scores(
                str(input_file), str(dest_report_path)
            )

        generate_cost_report(str(input_file))

        logger.info("Script completed successfully!")
        if not args.skip_evaluation and dest_report_path is not None:
            print(json.dumps({"report_json": str(dest_report_path)}))
        else:
            print(json.dumps({"report_json": ""}))

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
