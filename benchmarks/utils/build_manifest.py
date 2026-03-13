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
    cumulative_remote_check_seconds: float = 0.0
    cumulative_build_seconds: float = 0.0
    cumulative_post_build_seconds: float = 0.0
    cumulative_sdk_build_context_seconds: float = 0.0
    cumulative_sdk_buildx_wall_clock_seconds: float = 0.0
    cumulative_sdk_cleanup_seconds: float = 0.0
    cumulative_sdk_cache_import_seconds: float = 0.0
    cumulative_sdk_cache_export_seconds: float = 0.0
    cumulative_sdk_image_export_seconds: float = 0.0
    cumulative_sdk_push_layers_seconds: float = 0.0
    cumulative_sdk_export_manifest_seconds: float = 0.0
    cumulative_sdk_cache_import_misses: int = 0
    cumulative_sdk_cached_steps: int = 0
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


def _sum_float_field(records: list[dict[str, Any]], field_name: str) -> float:
    total = 0.0
    for record in records:
        value = _normalize_float(record.get(field_name))
        if value is not None:
            total += value
    return total


def _sum_int_field(records: list[dict[str, Any]], field_name: str) -> int:
    total = 0
    for record in records:
        value = record.get(field_name)
        if isinstance(value, bool):
            total += int(value)
        elif isinstance(value, int):
            total += value
        elif isinstance(value, float):
            total += int(value)
        elif isinstance(value, str) and value:
            try:
                total += int(value)
            except ValueError:
                continue
    return total


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
        cumulative_remote_check_seconds=_sum_float_field(
            records, "remote_check_seconds"
        ),
        cumulative_build_seconds=_sum_float_field(records, "build_seconds"),
        cumulative_post_build_seconds=_sum_float_field(records, "post_build_seconds"),
        cumulative_sdk_build_context_seconds=_sum_float_field(
            records, "sdk_build_context_seconds"
        ),
        cumulative_sdk_buildx_wall_clock_seconds=_sum_float_field(
            records, "sdk_buildx_wall_clock_seconds"
        ),
        cumulative_sdk_cleanup_seconds=_sum_float_field(records, "sdk_cleanup_seconds"),
        cumulative_sdk_cache_import_seconds=_sum_float_field(
            records, "sdk_cache_import_seconds"
        ),
        cumulative_sdk_cache_export_seconds=_sum_float_field(
            records, "sdk_cache_export_seconds"
        ),
        cumulative_sdk_image_export_seconds=_sum_float_field(
            records, "sdk_image_export_seconds"
        ),
        cumulative_sdk_push_layers_seconds=_sum_float_field(
            records, "sdk_push_layers_seconds"
        ),
        cumulative_sdk_export_manifest_seconds=_sum_float_field(
            records, "sdk_export_manifest_seconds"
        ),
        cumulative_sdk_cache_import_misses=_sum_int_field(
            records, "sdk_cache_import_miss_count"
        ),
        cumulative_sdk_cached_steps=_sum_int_field(records, "sdk_cached_step_count"),
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


def load_eval_env_summary(build_root: Path) -> dict[str, Any] | None:
    # SWT-Bench emits an additional eval-env summary alongside the shared
    # manifest-based image telemetry because its prebaked eval env build uses a
    # separate workflow path and artifact shape from the standard image builders.
    summary_file = next(build_root.rglob("eval-env-summary.json"), None)
    if summary_file is None:
        return None
    return json.loads(summary_file.read_text(encoding="utf-8"))


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

    phase_lines = [
        ("Remote Checks", summary.cumulative_remote_check_seconds),
        ("Build Wrapper Time", summary.cumulative_build_seconds),
        ("Post-Build Hooks", summary.cumulative_post_build_seconds),
        ("SDK Build Context", summary.cumulative_sdk_build_context_seconds),
        ("SDK Buildx Wall Clock", summary.cumulative_sdk_buildx_wall_clock_seconds),
        ("SDK Cache Imports", summary.cumulative_sdk_cache_import_seconds),
        ("SDK Cache Exports", summary.cumulative_sdk_cache_export_seconds),
        ("SDK Image Export", summary.cumulative_sdk_image_export_seconds),
        (
            "SDK Push Layers",
            summary.cumulative_sdk_push_layers_seconds,
        ),
        (
            "SDK Manifest Export",
            summary.cumulative_sdk_export_manifest_seconds,
        ),
    ]
    if (
        any(value > 0 for _, value in phase_lines)
        or summary.cumulative_sdk_cached_steps
    ):
        lines.extend(["", "### Phase Totals", ""])
        for label, value in phase_lines:
            if value > 0:
                lines.append(f"- **{label}:** {format_duration(value)}")
        if summary.cumulative_sdk_cache_import_misses:
            lines.append(
                f"- **SDK Cache Import Misses:** {summary.cumulative_sdk_cache_import_misses}"
            )
        if summary.cumulative_sdk_cached_steps:
            lines.append(
                f"- **SDK Cached Steps:** {summary.cumulative_sdk_cached_steps}"
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


def render_eval_env_summary_markdown(
    data: dict[str, Any], title: str = "SWT-Bench Eval Env Build Summary"
) -> str:
    summary = data.get("summary", {})
    lines = [
        f"## {title}",
        "",
        f"**Base Images Built:** {summary.get('built_base_images', 0)}",
        f"**Base Images Skipped:** {summary.get('skipped_base_images', 0)}",
        f"**Env Images Built:** {summary.get('built_env_images', 0)}",
        f"**Env Images Skipped:** {summary.get('skipped_env_images', 0)}",
    ]

    selected_env_instances = summary.get("selected_env_instances")
    if selected_env_instances is not None:
        lines.append(f"**Selected Env Instances:** {selected_env_instances}")

    lines.extend(
        [
            f"**Wall Clock:** {format_duration(_normalize_float(summary.get('wall_clock_seconds')))}",
            f"**Env Build Time:** {format_duration(_normalize_float(summary.get('env_build_seconds')))}",
            f"**Push Time:** {format_duration(_normalize_float(summary.get('push_seconds')))}",
        ]
    )

    batches = data.get("summary", {}).get("batches", [])
    if isinstance(batches, list) and batches:
        lines.extend(["", "### Eval Env Batches", ""])
        for batch in batches:
            if not isinstance(batch, dict):
                continue
            batch_index = batch.get("batch_index", "?")
            batch_size = batch.get("batch_size", "?")
            attempt_count = batch.get("attempt_count", "?")
            instance_count = batch.get("instance_count")
            duration = format_duration(_normalize_float(batch.get("duration_seconds")))
            line = (
                f"- Batch {batch_index}: {batch_size} unique images in {duration} "
                f"(attempts={attempt_count})"
            )
            if instance_count is not None:
                line += f", selected_instances={instance_count}"
            lines.append(line)

    return "\n".join(lines)
