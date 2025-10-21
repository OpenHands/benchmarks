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
import subprocess
import sys
from pathlib import Path

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


def install_swebench() -> None:
    """Install SWE-Bench if not already available."""
    import importlib.util

    if importlib.util.find_spec("swebench") is not None:
        logger.info("SWE-Bench is already installed")
        return

    logger.info("Installing SWE-Bench...")
    try:
        # Install SWE-Bench using uv
        subprocess.run(
            ["uv", "add", "swebench"], check=True, capture_output=True, text=True
        )
        logger.info("SWE-Bench installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install SWE-Bench: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise


def run_swebench_evaluation(
    predictions_file: str, dataset: str = "princeton-nlp/SWE-bench_Verified"
) -> None:
    """
    Run SWE-Bench evaluation on the predictions file.

    Args:
        predictions_file: Path to the SWE-Bench format predictions file
        dataset: SWE-Bench dataset to evaluate against
    """
    logger.info(f"Running SWE-Bench evaluation on {predictions_file}")

    try:
        # Get the directory of the predictions file
        predictions_path = Path(predictions_file)
        predictions_dir = predictions_path.parent
        predictions_filename = predictions_path.name

        # Run SWE-Bench evaluation using global python (not UV environment)
        # since swebench is installed globally
        cmd = [
            "/usr/bin/python3",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset,
            "--predictions_path",
            predictions_filename,
            "--max_workers",
            "5",
            "--run_id",
            f"eval_{predictions_path.stem}",
        ]

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
    uv run swebench-eval output.jsonl --model-name "MyModel-v1.0"
        """,
    )

    parser.add_argument("input_file", help="Path to the OpenHands output.jsonl file")

    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Verified",
        help="SWE-Bench dataset to evaluate against "
        "(default: princeton-nlp/SWE-bench_Verified)",
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
        "--install-swebench",
        action="store_true",
        help="Install SWE-Bench before running evaluation",
    )

    parser.add_argument(
        "--model-name",
        default="OpenHands",
        help="Model name to use in the model_name_or_path field (default: OpenHands)",
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
    logger.info(f"Model name: {args.model_name}")

    try:
        # Convert format
        convert_to_swebench_format(str(input_file), str(output_file), args.model_name)

        if not args.skip_evaluation:
            # Install SWE-Bench if requested
            if args.install_swebench:
                install_swebench()

            # Run evaluation
            run_swebench_evaluation(str(output_file), args.dataset)

        logger.info("Script completed successfully!")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
