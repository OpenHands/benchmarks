#!/usr/bin/env python3
"""
SWT-Bench Evaluation Script

This script converts OpenHands output.jsonl format to SWT-Bench prediction format
and runs the SWT-Bench evaluation.

Usage:
    uv run swtbench-eval <path_to_output.jsonl>
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.utils.patch_utils import remove_files_from_patch
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


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


def _format_id_preview(instance_ids: list[str], limit: int = 10) -> str:
    if len(instance_ids) <= limit:
        return ", ".join(instance_ids)
    return f"{', '.join(instance_ids[:limit])} ... (+{len(instance_ids) - limit} more)"


def _write_swtbench_sitecustomize(
    swt_bench_dir: Path, timing_file: Path
) -> Path:
    """
    Emit a sitecustomize.py into the SWT-Bench clone that records dataset timing.
    """
    sitecustomize_path = swt_bench_dir / "sitecustomize.py"
    content = f"""import json
import os
import time
from datetime import datetime, timezone

TIMING_FILE = os.environ.get("SWT_BENCH_TIMING_FILE")


def _append_timing(event, start_ts, end_ts, duration, extra=None):
    if not TIMING_FILE:
        return
    record = {{
        "event": event,
        "start_time": start_ts,
        "end_time": end_ts,
        "duration_seconds": duration,
    }}
    if extra:
        record.update(extra)
    try:
        with open(TIMING_FILE, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record) + "\\n")
    except Exception:
        # Best-effort timing; do not break evaluation
        pass


if TIMING_FILE:
    try:
        from src import dataset as _dataset

        _original_get_dataset_from_preds = _dataset.get_dataset_from_preds

        def _timed_get_dataset_from_preds(*args, **kwargs):
            start = time.perf_counter()
            start_ts = datetime.now(timezone.utc).isoformat()
            result = _original_get_dataset_from_preds(*args, **kwargs)
            end_ts = datetime.now(timezone.utc).isoformat()
            duration = time.perf_counter() - start

            instance_ids = None
            if len(args) > 2:
                instance_ids = args[2]
            else:
                instance_ids = kwargs.get("instance_ids")

            extra = {{
                "dataset_name": args[0] if len(args) > 0 else kwargs.get("dataset_name"),
                "split": args[1] if len(args) > 1 else kwargs.get("split"),
                "instance_ids": list(instance_ids) if instance_ids is not None else None,
                "run_id": args[4] if len(args) > 4 else kwargs.get("run_id"),
                "is_swt": args[5] if len(args) > 5 else kwargs.get("is_swt"),
                "filter_swt": args[6] if len(args) > 6 else kwargs.get("filter_swt"),
                "dataset_size": len(result) if hasattr(result, "__len__") else None,
            }}
            _append_timing("dataset_load", start_ts, end_ts, duration, extra)
            return result

        _dataset.get_dataset_from_preds = _timed_get_dataset_from_preds
    except Exception:
        pass
"""
    sitecustomize_path.write_text(content)
    return sitecustomize_path


def _log_timing_file(timing_file: Path) -> None:
    if not timing_file.exists():
        logger.warning("SWT-Bench timing file not found: %s", timing_file)
        return

    try:
        records = [
            json.loads(line)
            for line in timing_file.read_text().splitlines()
            if line.strip()
        ]
    except Exception as exc:  # pragma: no cover - best-effort logging
        logger.warning("Failed to read timing file %s: %s", timing_file, exc)
        return

    if not records:
        logger.info("No timing records captured in %s", timing_file)
        return

    for record in records:
        if record.get("event") != "dataset_load":
            continue
        ids = record.get("instance_ids") or []
        id_summary = _format_id_preview(ids, limit=5) if isinstance(ids, list) else ids
        logger.info(
            "Dataset load timing: start=%s end=%s duration=%.2fs size=%s "
            "is_swt=%s filter_swt=%s ids=%s",
            record.get("start_time"),
            record.get("end_time"),
            record.get("duration_seconds"),
            record.get("dataset_size"),
            record.get("is_swt"),
            record.get("filter_swt"),
            id_summary,
        )


def _build_run_id(predictions_path: Path) -> str:
    return f"eval_{predictions_path.stem}"


def _patch_swtbench_init(swt_bench_dir: Path) -> None:
    """Remove circular import in swt-bench src/__init__.py that pulls src.main."""
    init_path = swt_bench_dir / "src" / "__init__.py"
    if not init_path.exists():
        return

    text = init_path.read_text()
    marker = "from src.main import"
    if marker not in text:
        return

    cleaned = []
    for line in text.splitlines():
        if marker in line:
            cleaned.append("# " + line + "  # removed to avoid circular import")
        else:
            cleaned.append(line)
    init_path.write_text("\n".join(cleaned))
    logger.info("Patched swt-bench __init__.py to skip src.main import")


def _write_placeholder_report(
    report_path: Path,
    input_file: Path,
    dataset: str,
    model_name: str,
    converted_count: int,
    error_count: int,
    reason: str,
) -> None:
    """
    Emit a minimal report.json when we skip SWT-Bench evaluation.
    """
    report = {
        "input_file": input_file.name,
        "dataset": dataset,
        "model_name": model_name,
        "generated_at": _utcnow(),
        "metrics": {
            "total": converted_count,
            "success": 0,
            "success_rate": 0.0,
            "errors": error_count,
        },
        "resolved_ids": [],
        "unresolved_ids": [],
        "notes": reason,
    }
    report_path.write_text(json.dumps(report, indent=4))
    logger.info("Wrote placeholder report: %s", report_path)


def convert_to_swtbench_format(
    input_file: str, output_file: str, model_name: str = "OpenHands"
) -> tuple[int, int]:
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
        logger.warning("No valid entries converted from %s", input_file)

    return converted_count, error_count


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
    overall_start = time.perf_counter()
    logger.info(
        "Running SWT-Bench evaluation on %s (start: %s)", predictions_file, _utcnow()
    )

    try:
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
            _patch_swtbench_init(swt_bench_dir)
        else:
            _patch_swtbench_init(swt_bench_dir)

        # Get the directory and filename of the predictions file
        predictions_path = Path(predictions_file).resolve()
        predictions_filename = predictions_path.name
        run_id = _build_run_id(predictions_path)
        timing_file = swt_bench_dir / "evaluation_results" / f"{run_id}_timing.jsonl"
        timing_file.parent.mkdir(parents=True, exist_ok=True)
        if timing_file.exists():
            timing_file.unlink()

        # Copy predictions file to swt-bench directory
        swt_predictions_file = swt_bench_dir / predictions_filename
        shutil.copy2(predictions_file, swt_predictions_file)

        prediction_instance_ids = _load_prediction_instance_ids(predictions_path)
        if not prediction_instance_ids:
            raise ValueError(
                f"No instance IDs found in predictions file: {predictions_path}"
            )
        logger.info(
            "Prediction instance IDs (%s): %s",
            len(prediction_instance_ids),
            _format_id_preview(prediction_instance_ids),
        )

        sitecustomize_path = _write_swtbench_sitecustomize(swt_bench_dir, timing_file)

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
        existing_pythonpath = env.get("PYTHONPATH")
        pythonpath_entries = [str(swt_bench_dir)]
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        env["SWT_BENCH_TIMING_FILE"] = str(timing_file)

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
            run_id,
        ]
        cmd.extend(["--instance_ids", *prediction_instance_ids])

        logger.info(f"Using Python executable: {python_executable}")
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Working directory: {swt_bench_dir}")
        logger.info(f"PYTHONPATH: {env['PYTHONPATH']}")
        logger.info("SWT-Bench timing hooks -> sitecustomize: %s", sitecustomize_path)
        logger.info("Timing file: %s", timing_file)
        logger.info("SWT-Bench run start: %s", _utcnow())
        logger.info("SWT-Bench evaluation output:")
        print("-" * 80)

        # Stream output directly to console, running from swt-bench directory
        run_start = time.perf_counter()
        result = subprocess.run(cmd, text=True, cwd=swt_bench_dir, env=env)

        print("-" * 80)
        if result.returncode == 0:
            logger.info("SWT-Bench evaluation completed successfully")
        else:
            logger.error(
                f"SWT-Bench evaluation failed with return code {result.returncode}"
            )
            raise subprocess.CalledProcessError(result.returncode, cmd)

        _log_timing_file(timing_file)
        run_duration = time.perf_counter() - run_start
        total_duration = time.perf_counter() - overall_start
        logger.info(
            "SWT-Bench run end: %s (duration: %.2f seconds)",
            _utcnow(),
            run_duration,
        )
        logger.info(
            "Total evaluation duration: %.2f seconds",
            total_duration,
        )
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
        converted_count, error_count = convert_to_swtbench_format(
            str(input_file), str(output_file), args.model_name
        )
        report_path = input_file.with_name("output.report.json")

        if converted_count == 0 or args.skip_evaluation:
            reason = (
                "No predictions converted to SWT-Bench format"
                if converted_count == 0
                else "Evaluation skipped by flag"
            )
            logger.warning(
                "%s; producing placeholder report. Source: %s, errors: %s",
                reason,
                input_file,
                error_count,
            )
            _write_placeholder_report(
                report_path,
                input_file,
                args.dataset,
                args.model_name,
                converted_count,
                error_count,
                reason,
            )
            update_report_with_submitted_instances(report_path, output_file)
        else:
            # Run evaluation
            run_swtbench_evaluation(str(output_file), args.dataset, args.workers)

            # Move SWT-Bench evaluation report to same folder as output.jsonl
            cache_dir = Path.home() / ".cache" / "openhands" / "swt-bench"
            swt_bench_dir = cache_dir / "swt-bench"
            report_dir = swt_bench_dir / "evaluation_results"
            run_id = _build_run_id(output_file)
            model_name_safe = args.model_name.replace("/", "__")
            report_file = report_dir / f"{model_name_safe}.{run_id}.json"

            target_dir = input_file.parent
            shutil.move(str(report_file), str(report_path))
            logger.info(f"Moved evaluation report to: {report_path}")
            update_report_with_submitted_instances(report_path, output_file)

        # Generate cost report as final step
        generate_cost_report(str(input_file))

        logger.info("Script completed successfully!")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
