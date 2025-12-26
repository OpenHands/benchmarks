#!/usr/bin/env python3
"""
GAIA Evaluation Report Script

This script reads the output.jsonl produced by gaia run_infer, aggregates the
precomputed test_result.score values, optionally merges *_errors.jsonl to count
incomplete/error instances, and writes a SWE-bench-style summary report
(output.report.json next to the input file). It does not run model inference or
re-score answers; it only summarizes existing results and generates a cost
report.

Usage:
    uv run gaia-eval <path_to_output.jsonl>
"""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.utils.report import SwebenchReport, load_jsonl
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


def _read_eval_limit(output_path: Path) -> int | None:
    metadata_path = output_path.parent / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.warning("Failed to read metadata.json: %s", e)
        return None
    eval_limit = metadata.get("eval_limit")
    if isinstance(eval_limit, int) and eval_limit > 0:
        return eval_limit
    return None


def _error_output_path(input_file: Path) -> Path:
    return Path(str(input_file).replace(".jsonl", "_errors.jsonl"))


def build_gaia_report(input_file: Path) -> SwebenchReport:
    output_rows = load_jsonl(input_file)
    error_rows: list[dict[str, object]] = []

    error_path = _error_output_path(input_file)
    if error_path.exists():
        error_rows = load_jsonl(error_path)

    completed_ids: list[str] = []
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    error_ids: list[str] = []

    seen_ids: set[str] = set()

    for row in output_rows:
        instance_id = row.get("instance_id")
        if not instance_id:
            logger.warning("Skipping row without instance_id")
            continue
        if instance_id in seen_ids:
            continue
        seen_ids.add(instance_id)

        if row.get("error"):
            error_ids.append(instance_id)
            continue

        completed_ids.append(instance_id)
        score = row.get("test_result", {}).get("score")
        if bool(score):
            resolved_ids.append(instance_id)
        else:
            unresolved_ids.append(instance_id)

    for row in error_rows:
        instance_id = row.get("instance_id")
        if not instance_id:
            logger.warning("Skipping error row without instance_id")
            continue
        if instance_id in seen_ids:
            continue
        seen_ids.add(instance_id)
        error_ids.append(instance_id)

    total_instances = _read_eval_limit(input_file)
    if total_instances is None:
        total_instances = len(completed_ids) + len(error_ids)

    submitted_ids = completed_ids + error_ids

    return SwebenchReport.from_ids(
        total_instances=total_instances,
        completed_ids=completed_ids,
        resolved_ids=resolved_ids,
        unresolved_ids=unresolved_ids,
        error_ids=error_ids,
        submitted_ids=submitted_ids,
        empty_patch_ids=[],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GAIA evaluation report from output.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run gaia-eval output.jsonl
    uv run gaia-eval /path/to/output.jsonl
    uv run gaia-eval output.jsonl --output-file report.json
        """,
    )
    parser.add_argument("input_file", help="Path to the GAIA output.jsonl file")
    parser.add_argument(
        "--output-file",
        help=(
            "Output file for evaluation report "
            "(default: input_file with .report.json extension)"
        ),
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Only print metrics, skip writing report file",
    )
    parser.add_argument(
        "--model-name",
        default="openhands",
        help="Model name for logging (default: openhands)",
    )
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error("Input file does not exist: %s", input_file)
        sys.exit(1)

    if input_file.suffix != ".jsonl":
        logger.warning("Input file does not have .jsonl extension: %s", input_file)

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = input_file.with_suffix(".report.json")

    logger.info("Input file: %s", input_file)
    logger.info("Output file: %s", output_file)
    logger.info("Model name: %s", args.model_name)

    try:
        report = build_gaia_report(input_file)
        logger.info(
            "Report summary: total=%d submitted=%d resolved=%d unresolved=%d errors=%d",
            report.total_instances,
            report.submitted_instances,
            report.resolved_instances,
            report.unresolved_instances,
            report.error_instances,
        )

        if not args.skip_report:
            report.save(output_file)
            logger.info("Report generated successfully")

        generate_cost_report(str(input_file))

    except Exception as e:
        logger.error("Failed to generate report: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
