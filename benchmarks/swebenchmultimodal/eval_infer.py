#!/usr/bin/env python3
"""
SWE-Bench Multimodal Evaluation Script

This script converts OpenHands output.jsonl format to SWE-Bench prediction format
and runs the SWE-Bench Multimodal evaluation.

Usage:
    uv run swebenchmultimodal-eval <path_to_output.jsonl>
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.utils.patch_utils import remove_files_from_patch
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


def convert_to_swebench_format(
    input_file: str, output_file: str, model_name: str = "OpenHands"
) -> None:
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
        "model_name_or_path": "OpenHands"
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
                setup_files = ["pyproject.toml", "tox.ini", "setup.py"]
                git_patch = remove_files_from_patch(git_patch, setup_files)

                # Create SWE-Bench format entry
                swebench_entry = {
                    "instance_id": instance_id,
                    "model_patch": git_patch,
                    "model_name_or_path": model_name,
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


def run_swebench_multimodal_evaluation(
    predictions_file: str,
    dataset: str = "princeton-nlp/SWE-bench_Multimodal",
    split: str = "dev",
    workers: str = "12",
    run_id: str | None = None,
) -> None:
    """
    Run SWE-Bench Multimodal evaluation on the predictions file.

    Args:
        predictions_file: Path to the SWE-Bench format predictions file
        dataset: SWE-Bench dataset to evaluate against
        split: Dataset split to use (default: dev)
        workers: Number of workers to use for evaluation
        run_id: Optional run ID for the evaluation
    """
    logger.info(f"Running SWE-Bench Multimodal evaluation on {predictions_file}")

    try:
        # Get the directory of the predictions file
        predictions_path = Path(predictions_file)
        predictions_dir = predictions_path.parent
        predictions_filename = predictions_path.name

        # Generate run_id if not provided
        if run_id is None:
            run_id = f"eval_{predictions_path.stem}"

        # Run SWE-Bench Multimodal evaluation using UV environment
        # The key difference from regular SWE-Bench is the --modal true flag
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset,
            "--split",
            split,
            "--predictions_path",
            predictions_filename,
            "--max_workers",
            str(workers),
            "--modal",
            "true",
            "--run_id",
            run_id,
        ]

        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Working directory: {predictions_dir}")
        logger.info("SWE-Bench Multimodal evaluation output:")
        print("-" * 80)

        # Stream output directly to console, running from predictions file directory
        result = subprocess.run(cmd, text=True, cwd=predictions_dir)

        print("-" * 80)
        if result.returncode == 0:
            logger.info("SWE-Bench Multimodal evaluation completed successfully")
        else:
            logger.error(
                f"SWE-Bench Multimodal evaluation failed with return code {result.returncode}"
            )
            raise subprocess.CalledProcessError(result.returncode, cmd)

    except FileNotFoundError:
        logger.error(
            "SWE-Bench evaluation command not found. "
            "Make sure SWE-Bench is properly installed."
        )
        raise
    except Exception as e:
        logger.error(f"Error running SWE-Bench Multimodal evaluation: {e}")
        raise


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert OpenHands output to SWE-Bench format and run multimodal evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run swebenchmultimodal-eval output.jsonl
    uv run swebenchmultimodal-eval /path/to/output.jsonl --dataset princeton-nlp/SWE-bench_Multimodal
    uv run swebenchmultimodal-eval output.jsonl --model-name "MyModel-v1.0"
        """,
    )

    parser.add_argument("input_file", help="Path to the OpenHands output.jsonl file")

    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Multimodal",
        help="SWE-Bench dataset to evaluate against "
        "(default: princeton-nlp/SWE-bench_Multimodal)",
    )

    parser.add_argument(
        "--split",
        default="dev",
        help="Dataset split to use (default: dev)",
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
        "--model-name",
        default="openhands",
        help="Model name to use in the model_name_or_path field (default: openhands)",
    )

    parser.add_argument(
        "--workers",
        default="12",
        help="Number of workers to use when evaluating",
    )

    parser.add_argument(
        "--run-id",
        help="Run ID for the evaluation (default: eval_<output_filename>)",
    )

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
    logger.info(f"Split: {args.split}")
    logger.info(f"Model name: {args.model_name}")

    try:
        # Convert format
        convert_to_swebench_format(str(input_file), str(output_file), args.model_name)

        if not args.skip_evaluation:
            # Run multimodal evaluation
            run_swebench_multimodal_evaluation(
                str(output_file), args.dataset, args.split, args.workers, args.run_id
            )

            # Copy report.json to output.report.json for consistency with other benchmarks
            # SWE-Bench Multimodal creates report.json in the same directory as the predictions file
            report_path = output_file.parent / "report.json"
            dest_report_path = input_file.with_suffix(".report.json")

            if report_path.exists():
                shutil.copy(str(report_path), str(dest_report_path))
                logger.info(f"Copied report file to: {dest_report_path}")
            else:
                logger.warning(
                    f"Report file not found at expected location: {report_path}"
                )

        # Generate cost report as final step
        generate_cost_report(str(input_file))

        logger.info("Script completed successfully!")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
