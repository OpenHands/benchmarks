#!/usr/bin/env python3
"""SWE-Bench Pro evaluation script."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from benchmarks.swebenchpro import constants
from benchmarks.swebenchpro.config import EVAL_DEFAULTS
from benchmarks.utils.constants import MODEL_NAME_OR_PATH
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.laminar import LaminarService
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


def convert_to_swebenchpro_format(
    input_file: str,
    output_file: str,
    prefix: str = MODEL_NAME_OR_PATH,
) -> None:
    logger.info(
        "Converting %s to SWE-Bench Pro patch format: %s", input_file, output_file
    )

    converted_entries: list[dict[str, str]] = []
    error_count = 0

    with open(input_file, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                instance_id = data.get("instance_id")
                if not instance_id:
                    logger.warning("Line %s: Missing instance_id", line_num)
                    error_count += 1
                    continue

                test_result = data.get("test_result", {})
                git_patch = test_result.get("git_patch", "")
                if not git_patch:
                    logger.warning(
                        "Line %s: Missing or empty git_patch for %s",
                        line_num,
                        instance_id,
                    )
                    git_patch = ""

                converted_entries.append(
                    {
                        "instance_id": instance_id,
                        "patch": git_patch,
                        "prefix": prefix,
                    }
                )
            except json.JSONDecodeError as exc:
                logger.error("Line %s: Invalid JSON - %s", line_num, exc)
                error_count += 1
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Line %s: Unexpected error - %s", line_num, exc)
                error_count += 1

    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(converted_entries, outfile, indent=2)

    logger.info(
        "Conversion complete: %s entries converted, %s errors",
        len(converted_entries),
        error_count,
    )


def _extract_repo_archive(archive_bytes: bytes, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        archive.extractall(destination)
    top_level_dirs = {
        Path(member.name).parts[0]
        for member in members
        if member.name and not member.name.startswith("./")
    }
    if len(top_level_dirs) != 1:
        raise RuntimeError(
            f"Expected a single top-level directory in harness archive, found: {sorted(top_level_dirs)}"
        )
    return destination / next(iter(top_level_dirs))


def ensure_official_harness_checkout(
    cache_dir: str | None = None,
    archive_url: str = constants.OFFICIAL_HARNESS_ARCHIVE_URL,
    ref: str = constants.OFFICIAL_HARNESS_REF,
) -> Path:
    if cache_dir is None:
        cache_root = Path.home() / ".cache" / "openhands-benchmarks" / "swebenchpro"
    else:
        cache_root = Path(cache_dir)

    checkout_dir = cache_root / ref
    harness_script = checkout_dir / "swe_bench_pro_eval.py"
    if harness_script.exists():
        return checkout_dir

    logger.info("Downloading official SWE-Bench Pro harness from %s", archive_url)
    with urllib.request.urlopen(archive_url, timeout=60) as response:
        archive_bytes = response.read()

    with tempfile.TemporaryDirectory(
        dir=cache_root.parent if cache_root.parent.exists() else None
    ) as tmpdir:
        extracted_root = _extract_repo_archive(archive_bytes, Path(tmpdir))
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        if checkout_dir.exists():
            shutil.rmtree(checkout_dir)
        shutil.move(str(extracted_root), str(checkout_dir))

    return checkout_dir


def write_raw_sample_file(
    dataset: str,
    split: str,
    instance_ids: set[str],
    output_path: Path,
) -> None:
    df = get_dataset(dataset_name=dataset, split=split)
    if instance_ids:
        df = df[df["instance_id"].isin(sorted(instance_ids))]
    df.to_json(output_path, orient="records", lines=True)


def run_swebenchpro_evaluation(
    harness_dir: Path,
    raw_sample_path: Path,
    patch_path: Path,
    output_dir: Path,
    workers: int,
    dockerhub_username: str,
    use_local_docker: bool,
    block_network: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "swe_bench_pro_eval.py",
        "--raw_sample_path",
        str(raw_sample_path),
        "--patch_path",
        str(patch_path),
        "--output_dir",
        str(output_dir),
        "--scripts_dir",
        "run_scripts",
        "--num_workers",
        str(workers),
        "--dockerhub_username",
        dockerhub_username,
    ]
    if use_local_docker:
        cmd.append("--use_local_docker")
    if block_network:
        cmd.append("--block_network")

    logger.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=harness_dir, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)

    eval_results_path = output_dir / "eval_results.json"
    if not eval_results_path.exists():
        raise FileNotFoundError(f"Expected evaluation results at {eval_results_path}")
    return eval_results_path


def write_report(eval_results_path: Path, report_path: Path) -> dict[str, object]:
    with open(eval_results_path, "r", encoding="utf-8") as infile:
        eval_results = json.load(infile)

    resolved_ids = sorted(
        instance_id for instance_id, passed in eval_results.items() if passed
    )
    unresolved_ids = sorted(
        instance_id for instance_id, passed in eval_results.items() if not passed
    )
    submitted_ids = sorted(eval_results)

    report = {
        "total_instances": len(submitted_ids),
        "submitted_instances": len(submitted_ids),
        "completed_instances": len(submitted_ids),
        "incomplete_instances": 0,
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "error_instances": 0,
        "submitted_ids": submitted_ids,
        "completed_ids": submitted_ids,
        "incomplete_ids": [],
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "error_ids": [],
    }
    with open(report_path, "w", encoding="utf-8") as outfile:
        json.dump(report, outfile, indent=4)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OpenHands output to SWE-Bench Pro format and run evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", help="Path to the OpenHands output.jsonl file")
    parser.add_argument("--dataset", help="SWE-Bench Pro dataset to evaluate against")
    parser.add_argument(
        "--output-file",
        help="Output file for SWE-Bench Pro patch format (default: input_file with .swebenchpro.json extension)",
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
    parser.add_argument("--split", help="Dataset split to evaluate")
    parser.add_argument(
        "--dockerhub-username",
        help="Docker Hub username for official SWE-Bench Pro images",
    )
    parser.add_argument(
        "--official-harness-dir",
        help="Path to a local checkout of the official SWE-Bench Pro harness",
    )
    parser.add_argument(
        "--use-local-docker",
        dest="use_local_docker",
        action="store_true",
        help="Run evaluation with local Docker instead of Modal",
    )
    parser.add_argument(
        "--no-use-local-docker",
        dest="use_local_docker",
        action="store_false",
        help="Run evaluation with Modal instead of local Docker",
    )
    parser.add_argument(
        "--block-network",
        action="store_true",
        help="Block network access inside the evaluation containers",
    )
    parser.set_defaults(**EVAL_DEFAULTS)

    args = parser.parse_args()

    input_file = Path(args.input_file).resolve()
    if not input_file.exists():
        logger.error("Input file does not exist: %s", input_file)
        sys.exit(1)

    output_file = (
        Path(args.output_file).resolve()
        if args.output_file
        else input_file.with_suffix(".swebenchpro.json")
    )
    report_path = input_file.with_suffix(".report.json")

    try:
        convert_to_swebenchpro_format(str(input_file), str(output_file))

        if not args.skip_evaluation:
            harness_dir = (
                Path(args.official_harness_dir).resolve()
                if args.official_harness_dir
                else ensure_official_harness_checkout()
            )
            with open(output_file, "r", encoding="utf-8") as infile:
                patches = json.load(infile)
            instance_ids = {entry["instance_id"] for entry in patches}

            raw_sample_path = input_file.with_suffix(".swebenchpro.raw_samples.jsonl")
            write_raw_sample_file(
                dataset=args.dataset,
                split=args.split,
                instance_ids=instance_ids,
                output_path=raw_sample_path,
            )

            eval_output_dir = (
                input_file.with_suffix("").parent
                / f"{input_file.stem}.swebenchpro_eval"
            )
            eval_results_path = run_swebenchpro_evaluation(
                harness_dir=harness_dir,
                raw_sample_path=raw_sample_path,
                patch_path=output_file,
                output_dir=eval_output_dir,
                workers=args.workers,
                dockerhub_username=args.dockerhub_username,
                use_local_docker=args.use_local_docker,
                block_network=args.block_network,
            )
            write_report(eval_results_path, report_path)
            LaminarService.get().update_evaluation_scores(
                str(input_file), str(report_path)
            )

        generate_cost_report(str(input_file))
        logger.info("Script completed successfully!")
        print(
            json.dumps(
                {"report_json": "" if args.skip_evaluation else str(report_path)}
            )
        )
    except Exception as exc:
        logger.error("Script failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
