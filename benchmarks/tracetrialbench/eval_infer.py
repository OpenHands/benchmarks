#!/usr/bin/env python3
"""TraceTrial-Bench evaluation report generator.

Aggregates OpenHands ``output.jsonl`` entries produced by ``tracetrialbench-infer``
into the canonical OpenHands ``.report.json`` summary alongside TraceTrial-Bench
layered reliability metrics (Agent Score, Harness Macro, Clean Pass, per-dimension
breakdown across C/E/G/L/O/T/V, and failure attribution counts).

Usage:
    uv run tracetrialbench-eval <path/to/output.jsonl>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.tracetrialbench.config import EVAL_DEFAULTS
from benchmarks.utils.laminar import LaminarService
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)

DIMENSIONS: tuple[str, ...] = ("C", "E", "G", "L", "O", "T", "V")


def _dimension_from_instance_id(instance_id: str) -> str | None:
    """Extract the leading mechanism dimension letter (C/E/G/L/O/T/V) from task ID."""
    if not instance_id:
        return None
    lead = instance_id[0].upper()
    return lead if lead in DIMENSIONS else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def process_tracetrialbench_results(
    input_file: str,
    output_file: str,
    *,
    criterion: str = str(EVAL_DEFAULTS["resolution_criterion"]),
) -> dict[str, Any]:
    """Read ``output.jsonl`` and write a structured evaluation report."""
    if criterion not in {"agent_pass", "clean_pass"}:
        raise ValueError(
            f"Unsupported resolution criterion {criterion!r}; "
            "expected 'agent_pass' or 'clean_pass'"
        )

    logger.info("Processing TraceTrial-Bench results from %s", input_file)

    completed_ids: set[str] = set()
    incomplete_ids: set[str] = set()
    resolved_ids: set[str] = set()
    unresolved_ids: set[str] = set()
    error_ids: set[str] = set()

    total_cost_usd = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    agent_scores: list[float] = []
    harness_scores: list[float] = []
    overall_scores: list[float] = []
    agent_pass_count = 0
    clean_pass_count = 0

    dim_harness_scores: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
    dim_agent_scores: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
    attribution_counts: dict[str, int] = {}

    input_path = Path(input_file)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error(
                    "Invalid JSON on line %d of %s: %s", line_num, input_file, exc
                )
                continue

            instance_id = data.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                logger.warning("Line %d missing valid instance_id", line_num)
                continue

            error = data.get("error")
            test_result_raw = data.get("test_result")
            test_result: dict[str, Any] = (
                test_result_raw if isinstance(test_result_raw, dict) else {}
            )

            has_scorecard = (
                "overall_score" in test_result
                or "agent_pass" in test_result
                or "clean_pass" in test_result
            )

            if error and not has_scorecard:
                incomplete_ids.add(instance_id)
                error_ids.add(instance_id)
                continue

            completed_ids.add(instance_id)

            agent_pass = bool(test_result.get("agent_pass", False))
            clean_pass = bool(test_result.get("clean_pass", False))
            if agent_pass:
                agent_pass_count += 1
            if clean_pass:
                clean_pass_count += 1

            is_resolved = clean_pass if criterion == "clean_pass" else agent_pass
            if is_resolved:
                resolved_ids.add(instance_id)
            else:
                unresolved_ids.add(instance_id)

            a_score = _safe_float(test_result.get("agent_score_pct"), 0.0)
            h_score = _safe_float(test_result.get("harness_score_pct"), 0.0)
            o_score = _safe_float(test_result.get("overall_score"), 0.0)
            agent_scores.append(a_score)
            harness_scores.append(h_score)
            overall_scores.append(o_score)

            dim = test_result.get("dimension")
            dim_key = (
                dim.upper()
                if isinstance(dim, str) and dim.upper() in DIMENSIONS
                else _dimension_from_instance_id(instance_id)
            )
            if dim_key is not None:
                dim_harness_scores[dim_key].append(h_score)
                dim_agent_scores[dim_key].append(a_score)

            attribution_label = test_result.get("attribution")
            if isinstance(attribution_label, str) and attribution_label:
                attribution_counts[attribution_label] = (
                    attribution_counts.get(attribution_label, 0) + 1
                )

            metrics_raw = data.get("metrics")
            if isinstance(metrics_raw, dict):
                cost = metrics_raw.get("accumulated_cost")
                if cost is None:
                    cost = metrics_raw.get("total_cost_usd")
                total_cost_usd += _safe_float(cost, 0.0)

                token_usage = metrics_raw.get("accumulated_token_usage")
                if isinstance(token_usage, dict):
                    total_prompt_tokens += _safe_int(token_usage.get("prompt_tokens"))
                    total_completion_tokens += _safe_int(
                        token_usage.get("completion_tokens")
                    )
                else:
                    total_prompt_tokens += _safe_int(
                        metrics_raw.get("total_prompt_tokens")
                    )
                    total_completion_tokens += _safe_int(
                        metrics_raw.get("total_completion_tokens")
                    )

    error_path = input_path.with_name(f"{input_path.stem}_errors.jsonl")
    if error_path.exists():
        with error_path.open("r", encoding="utf-8") as error_handle:
            for line_num, raw_line in enumerate(error_handle, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    err_data: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.error(
                        "Invalid JSON on line %d of %s: %s", line_num, error_path, exc
                    )
                    continue
                err_id = err_data.get("instance_id")
                if not isinstance(err_id, str) or not err_id:
                    continue
                if err_id in completed_ids or err_id in incomplete_ids:
                    continue
                incomplete_ids.add(err_id)
                error_ids.add(err_id)

    submitted_ids = completed_ids | incomplete_ids
    n_completed = len(completed_ids)

    non_empty_dim_harness = [
        sum(vals) / len(vals) for vals in dim_harness_scores.values() if vals
    ]
    harness_macro = (
        sum(non_empty_dim_harness) / len(non_empty_dim_harness)
        if non_empty_dim_harness
        else 0.0
    )
    agent_score_mean = sum(agent_scores) / n_completed if n_completed else 0.0
    harness_micro_mean = sum(harness_scores) / n_completed if n_completed else 0.0
    overall_mean = (
        (agent_score_mean + harness_macro) / 2.0
        if non_empty_dim_harness
        else (sum(overall_scores) / n_completed if n_completed else 0.0)
    )

    per_dimension_summary: dict[str, dict[str, float | int]] = {}
    for dim in DIMENSIONS:
        h_vals = dim_harness_scores[dim]
        a_vals = dim_agent_scores[dim]
        if h_vals:
            per_dimension_summary[dim] = {
                "count": len(h_vals),
                "harness_score_pct": round(sum(h_vals) / len(h_vals), 2),
                "agent_score_pct": round(sum(a_vals) / len(a_vals), 2),
            }

    report: dict[str, Any] = {
        "total_instances": len(submitted_ids),
        "submitted_instances": len(submitted_ids),
        "completed_instances": n_completed,
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
        "resolution_criterion": criterion,
        "aggregate_metrics": {
            "total_cost_usd": round(total_cost_usd, 6),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
        },
        "tracetrial_metrics": {
            "agent_pass_rate": round(
                (agent_pass_count / n_completed * 100.0) if n_completed else 0.0, 2
            ),
            "clean_pass_rate": round(
                (clean_pass_count / n_completed * 100.0) if n_completed else 0.0, 2
            ),
            "agent_score_pct": round(agent_score_mean, 2),
            "harness_micro_pct": round(harness_micro_mean, 2),
            "harness_macro_pct": round(harness_macro, 2),
            "overall_score_pct": round(overall_mean, 2),
            "per_dimension": per_dimension_summary,
            "attribution_counts": dict(sorted(attribution_counts.items())),
        },
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_handle:
        json.dump(report, out_handle, indent=4)

    logger.info(
        "TraceTrial-Bench report written to %s: completed=%d, resolved=%d (%s), "
        "AgentScore=%.1f%%, HarnessMacro=%.1f%%, Clean=%.1f%%",
        output_file,
        n_completed,
        len(resolved_ids),
        criterion,
        report["tracetrial_metrics"]["agent_score_pct"],
        report["tracetrial_metrics"]["harness_macro_pct"],
        report["tracetrial_metrics"]["clean_pass_rate"],
    )
    return report


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``tracetrialbench-eval``."""
    parser = argparse.ArgumentParser(
        description="Aggregate TraceTrial-Bench output.jsonl into an evaluation report",
    )
    parser.add_argument(
        "input_file",
        help="Path to TraceTrial-Bench output.jsonl generated by tracetrialbench-infer",
    )
    parser.add_argument(
        "--output-file",
        help="Destination path for report JSON (default: <input_file>.report.json)",
    )
    parser.add_argument(
        "--criterion",
        choices=["agent_pass", "clean_pass"],
        default=str(EVAL_DEFAULTS["resolution_criterion"]),
        help=(
            "Criterion used to populate resolved_ids: 'agent_pass' (all task criteria "
            "pass, default) or 'clean_pass' (all task and harness criteria pass)."
        ),
    )

    args = parser.parse_args(argv)
    input_path = Path(args.input_file)
    if not input_path.is_file():
        logger.error("Input file does not exist: %s", input_path)
        sys.exit(1)

    output_path = (
        Path(args.output_file)
        if args.output_file
        else input_path.with_suffix(".report.json")
    )

    try:
        process_tracetrialbench_results(
            str(input_path),
            str(output_path),
            criterion=args.criterion,
        )
    except Exception as exc:
        logger.error("Failed to generate TraceTrial-Bench report: %s", exc)
        sys.exit(1)

    try:
        LaminarService.get().update_evaluation_scores(str(input_path), str(output_path))
    except Exception as exc:  # pragma: no cover - telemetry is non-critical
        logger.warning("Laminar update failed (non-critical): %s", exc)

    try:
        generate_cost_report(str(input_path))
    except Exception as exc:  # pragma: no cover - cost summary is non-critical
        logger.warning("Cost report generation failed (non-critical): %s", exc)

    print(json.dumps({"report_json": str(output_path)}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
