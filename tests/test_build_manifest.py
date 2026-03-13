from benchmarks.utils.build_manifest import (
    format_duration,
    render_build_summary_markdown,
    summarize_build_records,
)


def test_summarize_build_records_tracks_statuses_and_timings():
    records = [
        {
            "base_image": "repo/image-a",
            "status": "built",
            "attempt_count": 1,
            "duration_seconds": 30.0,
            "remote_check_seconds": 1.0,
            "build_seconds": 28.0,
            "post_build_seconds": 1.0,
            "sdk_build_context_seconds": 2.0,
            "sdk_buildx_wall_clock_seconds": 26.0,
            "sdk_cleanup_seconds": 0.5,
            "sdk_cache_import_seconds": 4.0,
            "sdk_cache_export_seconds": 8.0,
            "sdk_image_export_seconds": 10.0,
            "sdk_push_layers_seconds": 6.0,
            "sdk_export_manifest_seconds": 1.5,
            "sdk_cache_import_miss_count": 1,
            "sdk_cached_step_count": 2,
            "started_at": "2026-03-13T00:00:00+00:00",
            "finished_at": "2026-03-13T00:00:30+00:00",
            "tags": ["tag-a"],
        },
        {
            "base_image": "repo/image-b",
            "status": "skipped_remote_exists",
            "skip_reason": "remote_image_exists",
            "attempt_count": 1,
            "duration_seconds": 1.0,
            "started_at": "2026-03-13T00:00:31+00:00",
            "finished_at": "2026-03-13T00:00:32+00:00",
            "tags": ["tag-b"],
        },
        {
            "base_image": "repo/image-c",
            "status": "failed",
            "error": "boom",
            "attempt_count": 2,
            "duration_seconds": 60.0,
            "started_at": "2026-03-13T00:00:35+00:00",
            "finished_at": "2026-03-13T00:01:35+00:00",
            "tags": [],
        },
        {
            "base_image": "repo/image-d",
            "error": None,
            "duration_seconds": 20.0,
            "remote_check_seconds": 0.5,
            "build_seconds": 18.0,
            "post_build_seconds": 0.5,
            "sdk_build_context_seconds": 1.0,
            "sdk_buildx_wall_clock_seconds": 17.0,
            "sdk_cleanup_seconds": 0.4,
            "sdk_cache_import_seconds": 2.0,
            "sdk_cache_export_seconds": 3.0,
            "sdk_image_export_seconds": 5.0,
            "sdk_push_layers_seconds": 2.0,
            "sdk_export_manifest_seconds": 0.5,
            "sdk_cached_step_count": 1,
            "started_at": "2026-03-13T00:01:36+00:00",
            "finished_at": "2026-03-13T00:01:56+00:00",
            "tags": ["tag-d"],
        },
    ]

    summary = summarize_build_records(records, manifest_files=1, top_n=2)

    assert summary.total == 4
    assert summary.successful == 3
    assert summary.built == 2
    assert summary.skipped == 1
    assert summary.failed == 1
    assert summary.retried == 1
    assert summary.skip_reasons == {"remote_image_exists": 1}
    assert summary.status_counts["built"] == 2
    assert summary.status_counts["skipped_remote_exists"] == 1
    assert summary.status_counts["failed"] == 1
    assert summary.average_build_seconds == 25.0
    assert summary.median_build_seconds == 25.0
    assert summary.max_build_seconds == 30.0
    assert summary.wall_clock_seconds == 116.0
    assert summary.cumulative_duration_seconds == 111.0
    assert summary.cumulative_remote_check_seconds == 1.5
    assert summary.cumulative_build_seconds == 46.0
    assert summary.cumulative_post_build_seconds == 1.5
    assert summary.cumulative_sdk_build_context_seconds == 3.0
    assert summary.cumulative_sdk_buildx_wall_clock_seconds == 43.0
    assert summary.cumulative_sdk_cleanup_seconds == 0.9
    assert summary.cumulative_sdk_cache_import_seconds == 6.0
    assert summary.cumulative_sdk_cache_export_seconds == 11.0
    assert summary.cumulative_sdk_image_export_seconds == 15.0
    assert summary.cumulative_sdk_push_layers_seconds == 8.0
    assert summary.cumulative_sdk_export_manifest_seconds == 2.0
    assert summary.cumulative_sdk_cache_import_misses == 1
    assert summary.cumulative_sdk_cached_steps == 3
    assert [build.base_image for build in summary.slowest_builds] == [
        "repo/image-a",
        "repo/image-d",
    ]
    assert summary.failed_builds[0].base_image == "repo/image-c"


def test_render_build_summary_markdown_includes_profiling_fields():
    summary = summarize_build_records(
        [
            {
                "base_image": "repo/image-a",
                "status": "built",
                "attempt_count": 2,
                "duration_seconds": 42.0,
                "remote_check_seconds": 1.0,
                "build_seconds": 40.0,
                "sdk_cache_import_seconds": 5.0,
                "sdk_cache_export_seconds": 7.0,
                "sdk_image_export_seconds": 8.0,
                "sdk_push_layers_seconds": 4.0,
                "sdk_export_manifest_seconds": 1.0,
                "sdk_cache_import_miss_count": 2,
                "sdk_cached_step_count": 6,
                "started_at": "2026-03-13T00:00:00+00:00",
                "finished_at": "2026-03-13T00:00:42+00:00",
                "tags": ["tag-a"],
            }
        ],
        manifest_files=1,
    )

    markdown = render_build_summary_markdown(summary, "Example Build Summary")

    assert "## Example Build Summary" in markdown
    assert "**Built:** 1" in markdown
    assert "**Retried:** 1" in markdown
    assert "### Phase Totals" in markdown
    assert "**SDK Cache Imports:** 5s" in markdown
    assert "**SDK Cache Exports:** 7s" in markdown
    assert "**SDK Push Layers:** 4s" in markdown
    assert "**SDK Cache Import Misses:** 2" in markdown
    assert "### Slowest Built Images" in markdown
    assert "`repo/image-a`" in markdown
    assert "42s" in markdown


def test_format_duration_handles_empty_values():
    assert format_duration(None) == "n/a"
