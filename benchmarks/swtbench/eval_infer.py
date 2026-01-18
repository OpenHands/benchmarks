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
from time import monotonic

from benchmarks.swtbench.image_utils import (
    compute_required_images,
    ensure_swt_bench_repo,
)
from benchmarks.utils.laminar import LaminarService
from benchmarks.utils.patch_utils import (
    filter_patch_by_predicate,
    looks_like_test_file,
    remove_files_from_patch,
    remove_top_level_files,
)
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)

PREBAKED_REGISTRY = "ghcr.io/openhands/swtbench-eval"


def _load_prediction_instance_ids(predictions_file: Path) -> list[str]:
    instance_ids: list[str] = []
    seen = set()
    with predictions_file.open("r") as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Skipping invalid JSON in predictions file line %s: %s",
                    line_num,
                    e,
                )
                continue
            instance_id = data.get("instance_id")
            if not instance_id:
                logger.warning(
                    "Skipping predictions file line %s without instance_id",
                    line_num,
                )
                continue
            if instance_id in seen:
                continue
            seen.add(instance_id)
            instance_ids.append(instance_id)
    return instance_ids


def try_pull_prebaked_images(
    predictions_file: Path,
    dataset: str,
    split: str = "test",
    registry: str = PREBAKED_REGISTRY,
) -> None:
    """
    Best-effort pull of prebaked base/env images; no-op on failure.
    """
    try:
        base_images, env_images = compute_required_images(
            predictions_file,
            dataset,
            split,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Skipping prebaked image pull (compute failed): %s", exc)
        return

    tags = sorted(base_images | env_images)
    if not tags:
        logger.info("No prebaked images to pull (empty tag set)")
        return

    registry = registry.rstrip("/")
    for tag in tags:
        remote = f"{registry}/{tag}"
        logger.info("Attempting to pull prebaked image %s", remote)
        try:
            pull = subprocess.run(
                ["docker", "pull", remote],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            logger.warning("Docker not available; skipping prebaked image pull")
            return

        if pull.returncode != 0:
            logger.warning("Failed to pull %s: %s", remote, pull.stderr.strip())
            continue

        # Tag the remote image with the local name expected by the harness.
        tag_res = subprocess.run(
            ["docker", "tag", remote, tag],
            capture_output=True,
            text=True,
        )
        if tag_res.returncode != 0:
            logger.warning("Failed to tag %s as %s: %s", remote, tag, tag_res.stderr)
        else:
            logger.info("Pulled and tagged %s -> %s", remote, tag)


def update_report_with_submitted_instances(
    report_path: Path, predictions_path: Path
) -> None:
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found for update: {report_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found for update: {predictions_path}"
        )

    report = json.loads(report_path.read_text())
    submitted_ids = _load_prediction_instance_ids(predictions_path)
    report["submitted_instances"] = len(submitted_ids)
    report["submitted_ids"] = submitted_ids

    resolved_ids = report.get("resolved_ids")
    unresolved_ids = report.get("unresolved_ids")
    if isinstance(resolved_ids, list) and isinstance(unresolved_ids, list):
        completed_ids = sorted(set(resolved_ids) | set(unresolved_ids))
        report["completed_ids"] = completed_ids
        report["completed_instances"] = len(completed_ids)

    report_path.write_text(json.dumps(report, indent=4))
    logger.info(
        "Updated report with submitted_instances/submitted_ids: %s", report_path
    )


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
    skipped_no_tests = 0
    skipped_top_level_only = 0

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

                # postprocess git_patch: drop setup/config files
                setup_files = ["pyproject.toml", "tox.ini", "setup.py"]
                git_patch = remove_files_from_patch(git_patch, setup_files)

                # Drop top-level files (e.g., reproduction.py, helper scripts)
                git_patch, kept_after_top, dropped_top = remove_top_level_files(
                    git_patch
                )
                if kept_after_top == 0:
                    skipped_top_level_only += 1
                    logger.warning(
                        "Skipping %s: only top-level diffs present (dropped %s files)",
                        instance_id,
                        dropped_top,
                    )
                    continue

                # Keep only test-like files to avoid running helper/build files as tests.
                git_patch, kept, dropped = filter_patch_by_predicate(
                    git_patch,
                    lambda a, b: looks_like_test_file(a) or looks_like_test_file(b),
                )
                if kept == 0:
                    skipped_no_tests += 1
                    logger.warning(
                        "Skipping %s: no test-like diffs after filtering (dropped %s files)",
                        instance_id,
                        dropped,
                    )
                    continue

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
        "Conversion complete: %s entries converted, %s skipped (top-level only), %s skipped (no tests), %s errors",
        converted_count,
        skipped_top_level_only,
        skipped_no_tests,
        error_count,
    )

    if converted_count == 0:
        raise ValueError("No valid entries were converted")


def run_swtbench_evaluation(
    predictions_file: str,
    dataset: str = "princeton-nlp/SWE-bench_Verified",
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
        dataset: Dataset to evaluate against (SWE-bench by default for v0 parity)
        workers: Number of workers to use for evaluation
    """
    use_legacy = os.getenv("SWTBENCH_FORCE_CONDA", "").lower() in ("1", "true", "yes")
    mode = "legacy-conda" if use_legacy else "prebaked-images"
    logger.info("Running SWT-Bench evaluation on %s (mode=%s)", predictions_file, mode)

    try:
        swt_bench_dir = ensure_swt_bench_repo()

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

        is_swt_dataset = "swt-bench" in dataset.lower()
        cmd = [
            python_executable,
            "src/main.py",  # Run as script instead of module
            "--dataset_name",
            dataset,
            "--predictions_path",
            predictions_filename,
            "--max_workers",
            str(workers),
            "--run_id",
            f"eval_{predictions_path.stem}",
        ]
        if is_swt_dataset:
            cmd.append("--filter_swt")

        logger.info(f"Using Python executable: {python_executable}")
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Working directory: {swt_bench_dir}")
        logger.info(f"PYTHONPATH: {env['PYTHONPATH']}")
        logger.info("SWT-Bench evaluation output:")
        print("-" * 80)

        eval_start = monotonic()
        # Stream output directly to console, running from swt-bench directory
        result = subprocess.run(cmd, text=True, cwd=swt_bench_dir, env=env)
        eval_end = monotonic()

        print("-" * 80)
        if result.returncode == 0:
            logger.info(
                "SWT-Bench evaluation completed successfully in %.2fs",
                eval_end - eval_start,
            )
        else:
            logger.error(
                "SWT-Bench evaluation failed with return code %s after %.2fs",
                result.returncode,
                eval_end - eval_start,
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
        default="princeton-nlp/SWE-bench_Verified",
        help="Dataset to evaluate against "
        "(default: princeton-nlp/SWE-bench_Verified; set to an SWT-Bench dataset to use filter_swt)",
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

        # Default: use prebaked images; SWTbenCH_FORCE_CONDA opts into legacy flow.
        use_prebaked = os.getenv("SWTBENCH_FORCE_CONDA", "").lower() not in (
            "1",
            "true",
            "yes",
        )
        if use_prebaked:
            try_pull_prebaked_images(
                output_file,
                args.dataset,
            )
        else:
            logger.info(
                "SWTBENCH_FORCE_CONDA set; skipping prebaked image pull "
                "and using legacy (pre-mamba) evaluation flow"
            )

        if not args.skip_evaluation:
            eval_phase_start = monotonic()
            # Run evaluation
            run_swtbench_evaluation(str(output_file), args.dataset, args.workers)
            eval_phase_end = monotonic()

            cleanup_phase_start = monotonic()
            # Move SWT-Bench evaluation report to same folder as output.jsonl
            cache_dir = Path.home() / ".cache" / "openhands" / "swt-bench"
            swt_bench_dir = cache_dir / "swt-bench"
            report_dir = swt_bench_dir / "evaluation_results"
            run_id = f"eval_{output_file.stem}"
            model_name_safe = args.model_name.replace("/", "__")
            report_file = report_dir / f"{model_name_safe}.{run_id}.json"

            target_dir = input_file.parent
            target_file = target_dir / "output.report.json"
            shutil.move(str(report_file), str(target_file))
            logger.info(f"Moved evaluation report to: {target_file}")
            update_report_with_submitted_instances(target_file, output_file)

            # Update Laminar datapoints with evaluation scores
            LaminarService.get().update_evaluation_scores(
                str(input_file), str(target_file)
            )
            cleanup_phase_end = monotonic()

            logger.info(
                "Timing summary: swtbench_eval=%.2fs, report_move_and_metrics=%.2fs",
                eval_phase_end - eval_phase_start,
                cleanup_phase_end - cleanup_phase_start,
            )

        # Generate cost report as final step
        generate_cost_report(str(input_file))

        logger.info("Script completed successfully!")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
