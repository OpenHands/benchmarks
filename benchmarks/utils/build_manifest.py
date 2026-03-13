from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SlowBuild(BaseModel):
    base_image: str
    duration_seconds: float
    attempt_count: int = 1
    status: str


class FailedBuild(BaseModel):
    base_image: str
    error: str
    attempt_count: int = 1


class BuildManifestSummary(BaseModel):
    manifest_files: int = 0
    total: int = 0
    successful: int = 0
    built: int = 0
    skipped: int = 0
    failed: int = 0
    retried: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    wall_clock_seconds: float | None = None
    cumulative_duration_seconds: float = 0.0
    average_build_seconds: float | None = None
    median_build_seconds: float | None = None
    max_build_seconds: float | None = None
    status_counts: dict[str, int] = Field(default_factory=dict)
    skip_reasons: dict[str, int] = Field(default_factory=dict)
    slowest_builds: list[SlowBuild] = Field(default_factory=list)
    failed_builds: list[FailedBuild] = Field(default_factory=list)


def _normalize_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _record_status(record: dict[str, Any]) -> str:
    status = record.get("status")
    if isinstance(status, str) and status:
        return status
    if record.get("error") or not record.get("tags"):
        return "failed"
    return "built"


def load_manifest_records(build_root: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    manifest_files = sorted(build_root.rglob("manifest.jsonl"))
    records: list[dict[str, Any]] = []
    for manifest_file in manifest_files:
        for line in manifest_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
    return manifest_files, records


def summarize_build_records(
    records: list[dict[str, Any]], manifest_files: int = 0, top_n: int = 5
) -> BuildManifestSummary:
    status_counts: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    build_durations: list[float] = []
    started_at_candidates: list[datetime] = []
    finished_at_candidates: list[datetime] = []
    slowest_candidates: list[SlowBuild] = []
    failed_builds: list[FailedBuild] = []

    cumulative_duration = 0.0
    retried = 0

    for record in records:
        status = _record_status(record)
        status_counts[status] += 1

        attempt_count = int(record.get("attempt_count") or 1)
        if attempt_count > 1:
            retried += 1

        if status.startswith("skipped"):
            skip_reason = record.get("skip_reason") or status.removeprefix("skipped_")
            skip_reasons[str(skip_reason)] += 1

        started_at = _parse_datetime(record.get("started_at"))
        if started_at is not None:
            started_at_candidates.append(started_at)

        finished_at = _parse_datetime(record.get("finished_at"))
        if finished_at is not None:
            finished_at_candidates.append(finished_at)

        duration_seconds = _normalize_float(record.get("duration_seconds"))
        if duration_seconds is not None:
            cumulative_duration += duration_seconds

        if status == "built" and duration_seconds is not None:
            build_durations.append(duration_seconds)
            slowest_candidates.append(
                SlowBuild(
                    base_image=record.get("base_image", "unknown"),
                    duration_seconds=duration_seconds,
                    attempt_count=attempt_count,
                    status=status,
                )
            )

        if status == "failed":
            failed_builds.append(
                FailedBuild(
                    base_image=record.get("base_image", "unknown"),
                    error=record.get("error") or "No tags generated",
                    attempt_count=attempt_count,
                )
            )

    total = len(records)
    failed = status_counts.get("failed", 0)
    skipped = sum(
        count for status, count in status_counts.items() if status.startswith("skipped")
    )
    built = status_counts.get("built", 0)
    successful = total - failed

    started_at = (
        min(started_at_candidates).isoformat() if started_at_candidates else None
    )
    finished_at = (
        max(finished_at_candidates).isoformat() if finished_at_candidates else None
    )
    wall_clock_seconds = None
    if started_at_candidates and finished_at_candidates:
        wall_clock_seconds = (
            max(finished_at_candidates) - min(started_at_candidates)
        ).total_seconds()

    average_build_seconds = (
        statistics.mean(build_durations) if build_durations else None
    )
    median_build_seconds = (
        statistics.median(build_durations) if build_durations else None
    )
    max_build_seconds = max(build_durations) if build_durations else None
    slowest_builds = sorted(
        slowest_candidates, key=lambda build: build.duration_seconds, reverse=True
    )[:top_n]

    return BuildManifestSummary(
        manifest_files=manifest_files,
        total=total,
        successful=successful,
        built=built,
        skipped=skipped,
        failed=failed,
        retried=retried,
        started_at=started_at,
        finished_at=finished_at,
        wall_clock_seconds=wall_clock_seconds,
        cumulative_duration_seconds=cumulative_duration,
        average_build_seconds=average_build_seconds,
        median_build_seconds=median_build_seconds,
        max_build_seconds=max_build_seconds,
        status_counts=dict(status_counts),
        skip_reasons=dict(skip_reasons),
        slowest_builds=slowest_builds,
        failed_builds=failed_builds,
    )


def summarize_build_root(build_root: Path, top_n: int = 5) -> BuildManifestSummary:
    manifest_files, records = load_manifest_records(build_root)
    return summarize_build_records(
        records, manifest_files=len(manifest_files), top_n=top_n
    )


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def render_build_summary_markdown(
    summary: BuildManifestSummary, title: str, show_failures: bool = True
) -> str:
    lines = [f"## {title}", ""]
    if summary.manifest_files == 0:
        lines.append("❌ No `manifest.jsonl` files found.")
        return "\n".join(lines)

    lines.extend(
        [
            f"**Manifest Files:** {summary.manifest_files}",
            f"**Total Images:** {summary.total}",
            f"**Successful:** {summary.successful} ✅",
            f"**Built:** {summary.built} 🛠",
            f"**Skipped:** {summary.skipped} ⏭",
            f"**Failed:** {summary.failed} ❌",
            f"**Retried:** {summary.retried} 🔁",
            f"**Wall Clock:** {format_duration(summary.wall_clock_seconds)}",
            f"**Cumulative Image Time:** {format_duration(summary.cumulative_duration_seconds)}",
        ]
    )

    if summary.average_build_seconds is not None:
        lines.append(
            f"**Average Built Image Time:** {format_duration(summary.average_build_seconds)}"
        )
    if summary.median_build_seconds is not None:
        lines.append(
            f"**Median Built Image Time:** {format_duration(summary.median_build_seconds)}"
        )

    if summary.status_counts:
        lines.extend(["", "### Status Breakdown", ""])
        for status, count in sorted(summary.status_counts.items()):
            lines.append(f"- `{status}`: {count}")

    if summary.skip_reasons:
        lines.extend(["", "### Skip Reasons", ""])
        for reason, count in sorted(summary.skip_reasons.items()):
            lines.append(f"- `{reason}`: {count}")

    if summary.slowest_builds:
        lines.extend(["", "### Slowest Built Images", ""])
        for build in summary.slowest_builds:
            lines.append(
                f"- `{build.base_image}`: {format_duration(build.duration_seconds)} "
                f"(attempts={build.attempt_count})"
            )

    if show_failures and summary.failed_builds:
        lines.extend(["", "### Failed Builds", ""])
        for build in summary.failed_builds:
            lines.append(
                f"- `{build.base_image}`: {build.error} (attempts={build.attempt_count})"
            )

    return "\n".join(lines)
