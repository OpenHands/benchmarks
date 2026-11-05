"""Transparent JSON and Markdown reliability scorecards."""

from collections.abc import Iterable
from pathlib import Path

from benchmarks.reliability.artifacts import write_json
from benchmarks.reliability.models import (
    JsonObject,
    ReliabilityResult,
    as_json_object,
)


def write_scorecard_json(
    results: Iterable[ReliabilityResult],
    output_path: Path,
) -> None:
    """Write aggregate numerators and all per-run component results."""
    materialized = tuple(results)
    write_json(
        output_path,
        {
            "summary": _summary(materialized),
            "runs": [as_json_object(result) for result in materialized],
        },
    )


def write_scorecard_markdown(
    results: Iterable[ReliabilityResult],
    output_path: Path,
) -> None:
    """Write a human-readable scorecard without hiding component scores."""
    materialized = tuple(results)
    summary = _summary(materialized)
    lines = [
        "# Reliability scorecard",
        "",
        f"- Runs: {summary['runs']}",
        f"- Task passes: {summary['task_passes']}",
        f"- Completion/resume passes: {summary['completion_resume_passes']}",
        f"- No-duplicate-effect passes: {summary['no_duplicate_effect_passes']}",
        f"- Comparable overhead runs: {summary['overhead_comparable_runs']}",
        "",
        "| Run | Task | Resume | Effects | Overhead |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in materialized:
        lines.append(
            "| "
            f"`{result.run.run_id}` | "
            f"{_mark(result.task_passed)} | "
            f"{_mark(result.completion_resume.passed)} | "
            f"{_mark(result.no_duplicate_effect.passed)} | "
            f"{_mark(result.recovery_overhead.passed)} |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(results: tuple[ReliabilityResult, ...]) -> JsonObject:
    return {
        "runs": len(results),
        "task_passes": sum(result.task_passed for result in results),
        "completion_resume_passes": sum(
            result.completion_resume.passed for result in results
        ),
        "no_duplicate_effect_passes": sum(
            result.no_duplicate_effect.passed for result in results
        ),
        "overhead_comparable_runs": sum(
            result.recovery_overhead.passed for result in results
        ),
    }


def _mark(passed: bool) -> str:
    return "pass" if passed else "fail"
