#!/usr/bin/env python3
"""Generic Harbor evaluation report generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.utils.laminar import LaminarService
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


def _metric(
    data: dict[str, Any], test_result: dict[str, Any], key: str, default: Any
) -> Any:
    metrics = data.get("metrics", {})
    value = metrics.get(key)
    if value is not None:
        return value
    return test_result.get("final_metrics", {}).get(key, default)


def process_harbor_results(input_file: str, output_file: str) -> dict[str, Any]:
    completed_ids: set[str] = set()
    resolved_ids: set[str] = set()
    unresolved_ids: set[str] = set()
    incomplete_ids: set[str] = set()
    error_ids: set[str] = set()

    total_cost_usd = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    with open(input_file, encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error("Line %d: invalid JSON: %s", line_num, exc)
                continue

            instance_id = data.get("instance_id")
            if not instance_id:
                logger.warning("Line %d: missing instance_id", line_num)
                continue
            if instance_id in completed_ids or instance_id in incomplete_ids:
                logger.warning(
                    "Line %d: duplicate instance_id %s", line_num, instance_id
                )
                continue

            if data.get("error"):
                error_ids.add(instance_id)
                incomplete_ids.add(instance_id)
                continue

            test_result = data.get("test_result", {})
            completed_ids.add(instance_id)
            if test_result.get("passed") is True:
                resolved_ids.add(instance_id)
            else:
                unresolved_ids.add(instance_id)

            total_cost_usd += float(
                _metric(data, test_result, "total_cost_usd", 0.0) or 0.0
            )
            total_prompt_tokens += int(
                _metric(data, test_result, "total_prompt_tokens", 0) or 0
            )
            total_completion_tokens += int(
                _metric(data, test_result, "total_completion_tokens", 0) or 0
            )

    error_path = Path(input_file).with_name(f"{Path(input_file).stem}_errors.jsonl")
    if error_path.exists():
        with open(error_path, encoding="utf-8") as error_file:
            for line in error_file:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instance_id = data.get("instance_id")
                if instance_id and instance_id not in completed_ids:
                    incomplete_ids.add(instance_id)
                    error_ids.add(instance_id)

    submitted_ids = completed_ids | incomplete_ids
    report: dict[str, Any] = {
        "total_instances": len(submitted_ids),
        "submitted_instances": len(submitted_ids),
        "completed_instances": len(completed_ids),
        "incomplete_instances": len(incomplete_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "error_instances": len(error_ids),
        "submitted_ids": sorted(submitted_ids),
        "completed_ids": sorted(completed_ids),
        "incomplete_ids": sorted(incomplete_ids),
        "resolved_ids": sorted(resolved_ids),
        "unresolved_ids": sorted(unresolved_ids),
        "error_ids": sorted(error_ids),
        "aggregate_metrics": {
            "total_cost_usd": total_cost_usd,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
        },
    }

    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(report, outfile, indent=4)

    logger.info("Harbor report generated at %s", output_file)
    logger.info(
        "Resolved %d/%d completed instances", len(resolved_ids), len(completed_ids)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Process generic Harbor output.jsonl")
    parser.add_argument("input_file", help="Path to Harbor-converted output.jsonl")
    parser.add_argument("--output-file", help="Output report JSON path")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error("Input file does not exist: %s", input_file)
        sys.exit(1)

    output_file = (
        Path(args.output_file)
        if args.output_file
        else input_file.with_suffix(".report.json")
    )
    try:
        report = process_harbor_results(str(input_file), str(output_file))
        generate_cost_report(str(input_file))
    except Exception as exc:
        logger.error("Harbor evaluation failed: %s", exc)
        sys.exit(1)

    try:
        LaminarService.get().update_evaluation_scores(
            str(input_file), str(output_file)
        )
    except Exception as exc:
        logger.warning("Laminar telemetry reporting failed (non-fatal): %s", exc)

    print(json.dumps({"report_json": str(output_file)}))


if __name__ == "__main__":
    main()
