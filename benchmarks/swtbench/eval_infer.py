#!/usr/bin/env python3
"""
SWT-Bench Evaluation Script

This script converts OpenHands output.jsonl format to SWT-Bench prediction format
and runs the SWT-Bench evaluation.

Usage:
    uv run swtbench-eval <path_to_output.jsonl>
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.utils.patch_utils import remove_files_from_patch
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


def convert_to_swtbench_format(
    input_file: str, output_file: str, model_name: str = "OpenHands"
) -> None:
    """
    Convert OpenHands output.jsonl to SWT-Bench prediction format.

    OpenHands format:
    {
        "instance_id": "sympy__sympy-20590",
        "test_result": {
            "git_patch": "diff --git a/file.py b/file.py\n..."
        },
        "instruction": "...",
        "error": null,
        "history": [...]
    }

    SWT-Bench format:
    {
        "instance_id": "sympy__sympy-20590",
        "model_patch": "diff --git a/file.py b/file.py\n...",
        "model_name_or_path": "OpenHands"
    }
    """
    logger.info(f"Converting {input_file} to SWT-Bench format: {output_file}")

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

                # Create SWT-Bench format entry
                swtbench_entry = {
                    "instance_id": instance_id,
                    "model_patch": git_patch,
                    "model_name_or_path": model_name,
                }

                # Write to output file
                outfile.write(json.dumps(swtbench_entry) + "\n")
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


def ensure_docker_running() -> None:
    """Ensure Docker daemon is running, start it if needed."""
    try:
        # Check if Docker is already running
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            logger.info("Docker is already running")
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    logger.info("Starting Docker daemon...")
    try:
        # Start Docker daemon in the background (no sudo - running as root in container)
        subprocess.Popen(
            ["dockerd"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        # Wait for Docker to initialize
        import time
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            try:
                result = subprocess.run(
                    ["docker", "info"],
                    capture_output=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    logger.info("Docker daemon started successfully")
                    return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        raise RuntimeError("Docker daemon failed to start within 30 seconds")
    except Exception as e:
        logger.error(f"Failed to start Docker daemon: {e}")
        raise


def run_swtbench_evaluation(
    predictions_file: str,
    dataset: str = "eth-sri/SWT-bench_Verified_bm25_27k_zsp",
    workers: str = "12",
) -> None:
    """
    Run SWT-Bench evaluation on the predictions file.

    Note: The swt-bench package is included as a dependency in pyproject.toml
    to ensure all its dependencies are available, but the package itself is not
    properly structured for import. We use subprocess to run it from a cached
    clone since that's how the upstream package is designed to work.

    Args:
        predictions_file: Path to the SWT-Bench format predictions file
        dataset: SWT-Bench dataset to evaluate against
        workers: Number of workers to use for evaluation
    """
    logger.info(f"Running SWT-Bench evaluation on {predictions_file}")

    try:
        # Ensure Docker is running (required for SWT-Bench evaluation)
        ensure_docker_running()
        
        # Use a global cache directory for SWT-Bench source
        cache_dir = Path.home() / ".cache" / "openhands" / "swt-bench"
        swt_bench_dir = cache_dir / "swt-bench"

        # Clone SWT-Bench repository if it doesn't exist
        if not swt_bench_dir.exists():
            logger.info("Setting up SWT-Bench source in global cache...")
            cache_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Cloning SWT-Bench repository...")
            clone_cmd = [
                "git",
                "clone",
                "https://github.com/logic-star-ai/swt-bench.git",
                str(swt_bench_dir),
            ]
            result = subprocess.run(clone_cmd, text=True)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, clone_cmd)

            logger.info(f"SWT-Bench source installed at {swt_bench_dir}")

        # Get the directory and filename of the predictions file
        predictions_path = Path(predictions_file).resolve()
        predictions_filename = predictions_path.name

        # Copy predictions file to swt-bench directory
        swt_predictions_file = swt_bench_dir / predictions_filename
        shutil.copy2(predictions_file, swt_predictions_file)

        # Run SWT-Bench evaluation by running python directly from the swt-bench directory
        # but using the uv environment's python executable which has all dependencies
        benchmarks_dir = Path(__file__).parent.parent.parent

        # Get the python executable from the uv environment
        python_executable = subprocess.run(
            [
                "uv",
                "run",
                "--directory",
                str(benchmarks_dir),
                "python",
                "-c",
                "import sys; print(sys.executable)",
            ],
            capture_output=True,
            text=True,
            cwd=benchmarks_dir,
        ).stdout.strip()

        # Set up environment with PYTHONPATH to include swt-bench directory
        env = os.environ.copy()
        env["PYTHONPATH"] = str(swt_bench_dir)

        cmd = [
            python_executable,
            "src/main.py",  # Run as script instead of module
            "--dataset_name",
            dataset,
            "--predictions_path",
            predictions_filename,
            "--filter_swt",
            "--max_workers",
            str(workers),
            "--run_id",
            f"eval_{predictions_path.stem}",
        ]

        logger.info(f"Using Python executable: {python_executable}")
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Working directory: {swt_bench_dir}")
        logger.info(f"PYTHONPATH: {env['PYTHONPATH']}")
        logger.info("SWT-Bench evaluation output:")
        print("-" * 80)

        # Stream output directly to console, running from swt-bench directory
        result = subprocess.run(cmd, text=True, cwd=swt_bench_dir, env=env)

        print("-" * 80)
        if result.returncode == 0:
            logger.info("SWT-Bench evaluation completed successfully")
        else:
            logger.error(
                f"SWT-Bench evaluation failed with return code {result.returncode}"
            )
            raise subprocess.CalledProcessError(result.returncode, cmd)

    except FileNotFoundError:
        logger.error(
            "SWT-Bench evaluation command not found. "
            "Make sure git and python are available."
        )
        raise
    except Exception as e:
        logger.error(f"Error running SWT-Bench evaluation: {e}")
        raise


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert OpenHands output to SWT-Bench format and run evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run swtbench-eval output.jsonl
    uv run swtbench-eval /path/to/output.jsonl --dataset princeton-nlp/SWE-bench_Lite
    uv run swtbench-eval output.jsonl --model-name "MyModel-v1.0"
        """,
    )

    parser.add_argument("input_file", help="Path to the OpenHands output.jsonl file")

    parser.add_argument(
        "--dataset",
        default="eth-sri/SWT-bench_Verified_bm25_27k_zsp",
        help="SWT-Bench dataset to evaluate against "
        "(default: eth-sri/SWT-bench_Verified_bm25_27k_zsp)",
    )

    parser.add_argument(
        "--output-file",
        help="Output file for SWT-Bench format "
        "(default: input_file with .swtbench.jsonl extension)",
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only convert format, skip running evaluation",
    )

    parser.add_argument(
        "--model-name",
        default="OpenHands",
        help="Model name to use in the model_name_or_path field (default: OpenHands)",
    )

    parser.add_argument(
        "--workers",
        default="12",
        help="Number of workers to use when evaluating",
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
        output_file = input_file.with_suffix(".swtbench.jsonl")

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model name: {args.model_name}")

    try:
        # Convert format
        convert_to_swtbench_format(str(input_file), str(output_file), args.model_name)

        if not args.skip_evaluation:
            # Run evaluation
            run_swtbench_evaluation(str(output_file), args.dataset, args.workers)

        # Generate cost report as final step
        generate_cost_report(str(input_file))

        logger.info("Script completed successfully!")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
