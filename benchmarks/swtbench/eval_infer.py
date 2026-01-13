#!/usr/bin/env python3
"""
SWT-Bench Evaluation Script

This script converts OpenHands output.jsonl format to SWT-Bench prediction format
and runs the SWT-Bench evaluation.

Usage:
    uv run swtbench-eval <path_to_output.jsonl>
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from benchmarks.utils.laminar import LaminarService
from benchmarks.utils.patch_utils import remove_files_from_patch
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


def _now() -> str:
    """Return an ISO8601 UTC timestamp for logging."""
    return datetime.utcnow().isoformat() + "Z"


@contextlib.contextmanager
def _chdir(path: Path):
    """Temporarily change working directory."""
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


class TimingRecorder:
    """Lightweight timing helper to capture named durations."""

    def __init__(self) -> None:
        self._origin = time.perf_counter()
        self._events: dict[str, float] = {}

    def mark(self, name: str) -> None:
        self._events[name] = time.perf_counter()

    def elapsed(self, start: str, end: str | None = None) -> float:
        if end is None:
            end = start
            start = "start"
        start_time = self._events.get(start, self._origin)
        end_time = self._events.get(end, time.perf_counter())
        return end_time - start_time

    def summary(self) -> dict[str, float]:
        keys = sorted(self._events.keys(), key=self._events.get)
        if "start" not in self._events:
            keys.insert(0, "start")
            self._events["start"] = self._origin
        summary: dict[str, float] = {}
        for idx, key in enumerate(keys):
            next_key = keys[idx + 1] if idx + 1 < len(keys) else None
            start_time = self._events[key]
            end_time = self._events.get(next_key, time.perf_counter())
            summary[key] = end_time - start_time
        return summary


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


def run_swtbench_evaluation(
    predictions_file: str,
    dataset: str = "eth-sri/SWT-bench_Verified_bm25_27k_zsp",
    workers: str = "12",
    model_name: str = "OpenHands",
) -> None:
    """
    Run SWT-Bench evaluation on the predictions file.

    Args:
        predictions_file: Path to the SWT-Bench format predictions file
        dataset: SWT-Bench dataset to evaluate against
        workers: Number of workers to use for evaluation
        model_name: Model name stored in the predictions file/report
    """
    logger.info("Running SWT-Bench evaluation on %s", predictions_file)

    timers = TimingRecorder()
    timers.mark("start")

    cache_dir = Path(
        os.getenv(
            "SWT_BENCH_CACHE_DIR",
            Path.home() / ".cache" / "openhands" / "swt-bench",
        )
    )
    repo_override = os.getenv("SWT_BENCH_REPO_PATH")
    repo_candidates = [
        Path(p) for p in ([repo_override] if repo_override else []) if p
    ]
    repo_candidates.append(Path("/opt/swt-bench"))
    repo_candidates.append(cache_dir / "swt-bench")

    hf_cache = Path(
        os.getenv(
            "HF_HOME",
            os.getenv("SWT_BENCH_HF_CACHE", cache_dir / "huggingface"),
        )
    )
    hf_datasets_cache = Path(
        os.getenv(
            "HF_DATASETS_CACHE",
            os.getenv("SWT_BENCH_HF_DATASETS_CACHE", hf_cache / "datasets"),
        )
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)
    hf_datasets_cache.mkdir(parents=True, exist_ok=True)

    swt_bench_dir = None
    for candidate in repo_candidates:
        if candidate and candidate.exists():
            swt_bench_dir = candidate
            break

    if swt_bench_dir is None:
        swt_bench_dir = repo_candidates[-1]
        logger.info(
            "SWT-Bench source not found; cloning into %s (started %s)",
            swt_bench_dir,
            _now(),
        )
        swt_bench_dir.parent.mkdir(parents=True, exist_ok=True)

        clone_cmd = [
            "git",
            "clone",
            "https://github.com/logic-star-ai/swt-bench.git",
            str(swt_bench_dir),
        ]
        result = subprocess.run(clone_cmd, text=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, clone_cmd)

        logger.info("SWT-Bench source installed at %s", swt_bench_dir)
    else:
        logger.info("Using existing SWT-Bench source at %s", swt_bench_dir)

    os.environ.setdefault("SWT_BENCH_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("SWT_BENCH_REPO_PATH", str(swt_bench_dir))
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_datasets_cache))

    predictions_path = Path(predictions_file).resolve()
    run_id = f"eval_{predictions_path.stem}"
    max_workers = int(workers) if isinstance(workers, str) else workers
    cache_level = os.getenv("SWT_BENCH_CACHE_LEVEL", "env")
    clean_images = os.getenv("SWT_BENCH_CLEAN_IMAGES", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    force_rebuild = os.getenv("SWT_BENCH_FORCE_REBUILD", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    build_mode = os.getenv("SWT_BENCH_BUILD_MODE", "api")

    dataset_durations: list[float] = []
    run_timings: dict[str, float] = {}
    run_timestamps: dict[str, str] = {}
    report_file: Path | None = None

    logger.info("HF_HOME=%s HF_DATASETS_CACHE=%s", hf_cache, hf_datasets_cache)
    logger.info(
        "SWT-Bench run_id=%s cache_level=%s clean_images=%s force_rebuild=%s",
        run_id,
        cache_level,
        clean_images,
        force_rebuild,
    )

    src_path = swt_bench_dir / "src"

    with _chdir(swt_bench_dir):
        # Ensure swt-bench sources (including src/run_evaluation.py) are importable
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        sys.path.insert(0, str(swt_bench_dir))
        try:
            import src.dataset as swt_dataset
            import src.main as swt_main
            import src.run_evaluation as swt_run_eval
        except Exception:
            logger.error("Failed to import swt-bench modules from %s", swt_bench_dir)
            raise

        original_get_dataset = swt_dataset.get_dataset_from_preds
        original_run_instances = swt_run_eval.run_instances

        def _timed_get_dataset_from_preds(*args, **kwargs):
            start = time.perf_counter()
            result = original_get_dataset(*args, **kwargs)
            duration = time.perf_counter() - start
            dataset_durations.append(duration)
            logger.info(
                "Dataset load/prep finished in %.2fs at %s (instances=%s)",
                duration,
                _now(),
                len(result) if result else 0,
            )
            return result

        def _timed_run_instances(*args, **kwargs):
            run_timings["start"] = time.perf_counter()
            run_timestamps["start"] = _now()
            try:
                return original_run_instances(*args, **kwargs)
            finally:
                run_timings["end"] = time.perf_counter()
                run_timestamps["end"] = _now()
                logger.info(
                    "SWT-Bench run_instances completed in %.2fs (start=%s end=%s)",
                    run_timings["end"] - run_timings["start"],
                    run_timestamps.get("start", "unknown"),
                    run_timestamps.get("end", "unknown"),
                )

        swt_dataset.get_dataset_from_preds = _timed_get_dataset_from_preds
        swt_run_eval.run_instances = _timed_run_instances

        timers.mark("swtbench_start")
        logger.info(
            "Starting SWT-Bench harness (run_id=%s) at %s with %s workers",
            run_id,
            _now(),
            max_workers,
        )

        try:
            swt_main.run(
                dataset_name=dataset,
                is_swt=False,
                split="test",
                instance_ids=None,
                predictions_path=str(predictions_path),
                compute_coverage=True,
                max_workers=max_workers,
                force_rebuild=force_rebuild,
                cache_level=cache_level,
                clean=clean_images,
                open_file_limit=4096,
                run_id=run_id,
                patch_types=["vanilla"],
                timeout=1800,
                filter_swt=True,
                build_mode=build_mode,
                skip_eval=False,
                exec_mode="unit_test",
                reproduction_script_name=None,
            )
        except Exception:
            logger.exception("SWT-Bench evaluation failed")
            raise
        finally:
            swt_dataset.get_dataset_from_preds = original_get_dataset
            swt_run_eval.run_instances = original_run_instances
            timers.mark("swtbench_end")

        report_dir = swt_bench_dir / "evaluation_results"
        model_name_safe = model_name.replace("/", "__")
        report_file = report_dir / f"{model_name_safe}.{run_id}.json"
        logger.info("Expected report at %s", report_file)

    timers.mark("end")

    if report_file and report_file.exists():
        logger.info("SWT-Bench evaluation completed successfully at %s", _now())
    else:
        logger.error("SWT-Bench evaluation finished without report output")
        raise FileNotFoundError(f"Report file not found: {report_file}")

    if dataset_durations:
        total_dataset_time = sum(dataset_durations)
        logger.info(
            "Dataset load/preprocessing time: %.2fs over %d call(s)",
            total_dataset_time,
            len(dataset_durations),
        )
    else:
        logger.info("Dataset load timing unavailable (no calls recorded)")

    if run_timings:
        logger.info(
            "SWT-Bench run_instances timing: start=%s end=%s duration=%.2fs",
            run_timestamps.get("start", "unknown"),
            run_timestamps.get("end", "unknown"),
            run_timings["end"] - run_timings["start"],
        )

    total_duration = timers.elapsed("start", "end")
    harness_duration = timers.elapsed("swtbench_start", "swtbench_end")
    logger.info(
        "SWT-Bench evaluation total duration: %.2fs (harness: %.2fs)",
        total_duration,
        harness_duration,
    )


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
            run_swtbench_evaluation(
                str(output_file),
                args.dataset,
                args.workers,
                args.model_name,
            )

            # Move SWT-Bench evaluation report to same folder as output.jsonl
            cache_dir = Path(
                os.getenv(
                    "SWT_BENCH_CACHE_DIR",
                    Path.home() / ".cache" / "openhands" / "swt-bench",
                )
            )
            swt_bench_dir = Path(
                os.getenv("SWT_BENCH_REPO_PATH", cache_dir / "swt-bench")
            )
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

        # Generate cost report as final step
        generate_cost_report(str(input_file))

        logger.info("Script completed successfully!")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
