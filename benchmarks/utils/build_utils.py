#!/usr/bin/env python3
"""
Shared utilities for batch building agent-server images.
"""

import argparse
import contextlib
import io
import os
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Callable, Literal

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from benchmarks.swebench.constants import TargetType
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.build_manifest import summarize_build_records
from benchmarks.utils.buildx_utils import (
    buildkit_disk_usage,
    maybe_prune_buildkit_cache,
    maybe_reset_buildkit,
)
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.image_utils import local_image_exists, remote_image_exists
from openhands.sdk import get_logger


logger = get_logger(__name__)


class BuildOutput(BaseModel):
    time: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    base_image: str
    tags: list[str]
    error: str | None = None
    log_path: str | None = None
    status: Literal["built", "skipped_remote_exists", "failed"] = "built"
    skip_reason: str | None = None
    attempt_count: int = 1
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    remote_check_seconds: float | None = None
    build_seconds: float | None = None
    post_build_seconds: float | None = None
    sdk_build_context_seconds: float | None = None
    sdk_buildx_wall_clock_seconds: float | None = None
    sdk_cleanup_seconds: float | None = None
    sdk_cache_import_seconds: float | None = None
    sdk_cache_import_miss_count: int | None = None
    sdk_cache_export_seconds: float | None = None
    sdk_image_export_seconds: float | None = None
    sdk_push_layers_seconds: float | None = None
    sdk_export_manifest_seconds: float | None = None
    sdk_cached_step_count: int | None = None


def run_docker_build_layer(
    dockerfile: Path | str,
    context: Path | str,
    tags: list[str],
    build_args: dict[str, str] | None = None,
    push: bool = False,
    platform: str = "linux/amd64",
    load: bool = True,
    no_cache: bool = False,
) -> BuildOutput:
    """
    Run docker buildx build to apply a custom layer on top of an existing image.

    This is a shared helper for building thin wrapper images (e.g., SWE-bench docutils/roman,
    GAIA MCP-precache, OpenAgentSafety local image).

    Args:
        dockerfile: Path to the Dockerfile to build.
        context: Path to the build context directory.
        tags: List of tags to apply to the built image.
        build_args: Optional dict of build arguments (e.g., {"SDK_IMAGE": "..."}).
        push: If True, push to registry via buildx. If False and load is True, load locally.
        platform: Target platform (default: linux/amd64).
        load: If True and push is False, load the image into local docker.
        no_cache: If True, pass --no-cache to disable layer cache.

    Returns:
        BuildOutput with tags on success, or error message on failure.
    """
    dockerfile_path = Path(dockerfile)
    context_path = Path(context)

    if not dockerfile_path.exists():
        return BuildOutput(
            base_image=str(dockerfile),
            tags=[],
            error=f"Dockerfile not found at {dockerfile_path}",
        )

    if not context_path.exists():
        return BuildOutput(
            base_image=str(context),
            tags=[],
            error=f"Build context not found at {context_path}",
        )

    # Build command
    cmd = ["docker", "buildx", "build", "--file", str(dockerfile_path)]

    # Add build arguments
    if build_args:
        for key, value in build_args.items():
            cmd.extend(["--build-arg", f"{key}={value}"])

    # Add tags
    for tag in tags:
        cmd.extend(["--tag", tag])

    # Add platform
    cmd.extend(["--platform", platform])

    # Push or load
    if push:
        cmd.append("--push")
    elif load:
        cmd.append("--load")

    # Add no-cache if requested
    if no_cache:
        cmd.append("--no-cache")

    # Add context path
    cmd.append(str(context_path))

    logger.info("Running docker build: %s", " ".join(cmd))

    # Run build with output capture
    proc = subprocess.run(cmd, text=True, capture_output=True)

    # Log output so it appears in capture_output logs when called from _build_with_logging
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    if proc.returncode != 0:
        error = (
            proc.stderr.strip()
            or proc.stdout.strip()
            or f"Docker build failed with exit code {proc.returncode}"
        )
        return BuildOutput(base_image=str(dockerfile), tags=[], error=error)

    return BuildOutput(base_image=str(dockerfile), tags=tags, error=None)


def _get_sdk_submodule_info() -> tuple[str, str, str]:
    """
    Get SDK version info from the vendor/software-agent-sdk submodule.

    Returns:
        tuple[str, str, str]: (git_ref, git_sha, sdk_version)
    """
    # Find the benchmarks repo root (where this file lives)
    benchmarks_root = Path(__file__).resolve().parent.parent.parent
    sdk_path = benchmarks_root / "vendor" / "software-agent-sdk"

    # Get submodule SHA directly from the checked-out submodule
    # This is more direct than parsing git submodule status output
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=sdk_path,
            capture_output=True,
            text=True,
            check=True,
        )
        git_sha = result.stdout.strip()
    except subprocess.CalledProcessError:
        logger.warning(
            "Failed to get SDK submodule SHA, using 'unknown'. "
            "Make sure submodules are initialized."
        )
        git_sha = "unknown"

    # Get submodule ref (current branch or HEAD)
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "-q", "--short", "HEAD"],
            cwd=sdk_path,
            capture_output=True,
            text=True,
            check=True,
        )
        git_ref = result.stdout.strip()
    except subprocess.CalledProcessError:
        git_ref = "unknown"

    # Get SDK version from pyproject.toml
    pyproject_path = sdk_path / "openhands-sdk" / "pyproject.toml"
    sdk_version = "unknown"
    try:
        if pyproject_path.exists():
            with pyproject_path.open("rb") as f:
                config = tomllib.load(f)
            sdk_version = config.get("project", {}).get("version", "unknown")
    except Exception as e:
        logger.warning(f"Failed to read SDK version from pyproject.toml: {e}")

    logger.info(
        f"SDK submodule info: ref={git_ref}, sha={git_sha[:7]}, version={sdk_version}"
    )
    return git_ref, git_sha, sdk_version


@contextlib.contextmanager
def capture_output(base_name: str, out_dir: Path):
    """
    Capture stdout/stderr during a block and stream them to:
      <out_dir>/<base_name>/build-<timestamp>.log

    Keeps redirect_* semantics; writes are realtime (line-buffered + flush).
    Yields the log_path.
    """
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    log_path = Path(out_dir) / base_name / f"build-{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # tell the user where we’re logging, without being swallowed by the redirect
    # (goes to the original stderr so it’s visible immediately)
    logger.info(f"Logging build output to {log_path}")

    # Open line-buffered so writes flush on newlines;
    # also wrap to hard-flush every write.
    f = log_path.open("w", encoding="utf-8", buffering=1)

    class _FlushOnWrite(io.TextIOBase):
        encoding = f.encoding

        def __init__(self, sink):
            self._sink = sink

        def write(self, s):
            n = self._sink.write(s)
            self._sink.flush()
            return n

        def flush(self):
            self._sink.flush()

        def fileno(self):
            # allow libs that try to detect fileno()
            return self._sink.fileno()

    sink = _FlushOnWrite(f)

    # Redirect stdout/stderr to the same realtime sink.
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):  # type: ignore[arg-type]
        try:
            yield log_path
        finally:
            # make sure everything is on disk
            sink.flush()
            f.close()


def get_build_parser() -> argparse.ArgumentParser:
    """Reuse benchmark parser and extend with build-related options."""
    parser = get_parser(add_llm_config=False)
    parser.description = "Script for build agent-server images."
    parser.add_argument(
        "--image",
        default=EVAL_AGENT_SERVER_IMAGE,
        help="Target repo/name for built image",
    )
    parser.add_argument(
        "--target",
        default="source-minimal",
        help="Build target (source | source-minimal | binary | binary-minimal)",
    )
    parser.add_argument(
        "--push", action="store_true", help="Push via buildx instead of load locally"
    )
    parser.add_argument(
        "--max-workers", type=int, default=1, help="Concurrent builds (be cautious)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List base images only, don’t build"
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Rebuild images even if matching remote tags already exist",
    )
    return parser


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _round_duration(seconds: float) -> float:
    return round(seconds, 3)


def _force_build_enabled(force_build: bool = False) -> bool:
    env_force_build = os.getenv("FORCE_BUILD", "0").lower() in ("1", "true", "yes")
    return force_build or env_force_build


def _apply_sdk_telemetry(output: BuildOutput, telemetry: object | None) -> BuildOutput:
    if telemetry is None:
        return output

    output.sdk_build_context_seconds = getattr(telemetry, "build_context_seconds", None)
    output.sdk_buildx_wall_clock_seconds = getattr(
        telemetry, "buildx_wall_clock_seconds", None
    )
    output.sdk_cleanup_seconds = getattr(telemetry, "cleanup_seconds", None)
    output.sdk_cache_import_seconds = getattr(telemetry, "cache_import_seconds", None)
    output.sdk_cache_import_miss_count = getattr(
        telemetry, "cache_import_miss_count", None
    )
    output.sdk_cache_export_seconds = getattr(telemetry, "cache_export_seconds", None)
    output.sdk_image_export_seconds = getattr(telemetry, "image_export_seconds", None)
    output.sdk_push_layers_seconds = getattr(telemetry, "push_layers_seconds", None)
    output.sdk_export_manifest_seconds = getattr(
        telemetry, "export_manifest_seconds", None
    )
    output.sdk_cached_step_count = getattr(telemetry, "cached_step_count", None)
    return output


def build_image(
    base_image: str,
    target_image: str,
    custom_tag: str,
    target: TargetType = "source-minimal",
    push: bool = False,
    force_build: bool = False,
) -> BuildOutput:
    # Importing here because openhands.agent_server.docker.build runs git checks
    # which fails when installed as a package outside the git repo
    from openhands.agent_server.docker.build import BuildOptions, build_with_telemetry

    # Get SDK info from submodule to ensure tags use the correct SDK SHA
    git_ref, git_sha, sdk_version = _get_sdk_submodule_info()
    remote_check_seconds = 0.0

    opts = BuildOptions(
        base_image=base_image,
        custom_tags=custom_tag,
        image=target_image,
        target=target,
        # SWE-Bench only supports linux/amd64 images
        platforms=["linux/amd64"],
        push=push,
        # Override git info to use SDK submodule info instead of benchmarks repo
        git_ref=git_ref,
        git_sha=git_sha,
        sdk_version=sdk_version,
    )
    if _force_build_enabled(force_build):
        logger.info(
            "FORCE_BUILD set, rebuilding remote image for %s even if it exists.",
            base_image,
        )
    else:
        for t in opts.all_tags:
            # Check if image exists or not
            remote_check_started = time.monotonic()
            exists = remote_image_exists(t)
            remote_check_seconds += time.monotonic() - remote_check_started
            if exists:
                logger.info("Image %s already exists. Skipping build.", t)
                return BuildOutput(
                    base_image=base_image,
                    tags=[t],
                    error=None,
                    status="skipped_remote_exists",
                    skip_reason="remote_image_exists",
                    remote_check_seconds=_round_duration(remote_check_seconds),
                    build_seconds=0.0,
                )
    build_started = time.monotonic()
    try:
        build_result = build_with_telemetry(opts)
        tags = build_result.tags
    except Exception as exc:
        return _apply_sdk_telemetry(
            BuildOutput(
                base_image=base_image,
                tags=[],
                error=repr(exc),
                status="failed",
                remote_check_seconds=_round_duration(remote_check_seconds),
                build_seconds=_round_duration(time.monotonic() - build_started),
            ),
            getattr(exc, "telemetry", None),
        )
    return _apply_sdk_telemetry(
        BuildOutput(
            base_image=base_image,
            tags=tags,
            error=None,
            status="built",
            remote_check_seconds=_round_duration(remote_check_seconds),
            build_seconds=_round_duration(time.monotonic() - build_started),
        ),
        build_result.telemetry,
    )


def ensure_local_image(
    agent_server_image: str,
    base_image: str,
    custom_tag: str,
    target: TargetType = "source-minimal",
) -> bool:
    """Build an agent-server image locally if it doesn't already exist.

    Returns True if a build occurred, False if the image already existed.
    Set FORCE_BUILD=1 to skip auto-detection and always rebuild.
    """
    force_build = _force_build_enabled()
    if not force_build and local_image_exists(agent_server_image):
        logger.info(f"Using pre-built image {agent_server_image}")
        return False

    if force_build:
        logger.info(f"FORCE_BUILD set, building image from {base_image}...")
    else:
        logger.info(f"Building image from {base_image}...")
    output = build_image(
        base_image=base_image,
        target_image=EVAL_AGENT_SERVER_IMAGE,
        custom_tag=custom_tag,
        target=target,
        push=False,
    )
    logger.info(f"Image build output: {output}")
    if output.error is not None:
        raise RuntimeError(f"Image build failed: {output.error}")
    if agent_server_image not in output.tags:
        raise RuntimeError(
            f"Built image tags {output.tags} do not include expected tag "
            f"{agent_server_image}"
        )
    return True


def _build_with_logging(
    log_dir: Path,
    base_image: str,
    target_image: str,
    custom_tag: str = "",
    target: TargetType = "source-minimal",
    push: bool = False,
    force_build: bool = False,
    max_retries: int = 3,
    post_build_fn: Callable[[BuildOutput, bool], BuildOutput] | None = None,
) -> BuildOutput:
    """
    Module-level function for building a single image with output capture.
    Must be at module level to be picklable for ProcessPoolExecutor.
    Automatically retries failed builds up to max_retries times.
    Timing fields on the returned BuildOutput are cumulative across all attempts:
    remote/build/post-build seconds are summed across retries, while
    duration_seconds is the overall wall-clock duration including retry sleeps.

    Args:
        custom_tag: Custom tag (already resolved) to pass to build_image.
        post_build_fn: Optional callback called after successful build.
            Receives (build_result, push) and returns modified BuildOutput.
            If it returns an error, the build is retried.
    """
    assert max_retries >= 1, "max_retries must be at least 1"
    overall_started_at = _utcnow_iso()
    overall_started_monotonic = time.monotonic()
    remote_check_total = 0.0
    build_total = 0.0
    post_build_total = 0.0
    final_result: BuildOutput | None = None
    attempts_used = 0

    for attempt in range(max_retries):
        attempts_used = attempt + 1
        with capture_output(base_image, log_dir) as log_path:
            if attempt > 0:
                logger.info(
                    f"Retrying build for {base_image} (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(2 + attempt * 2)
            logger.info(
                "Starting build for %s (attempt %d/%d)",
                base_image,
                attempt + 1,
                max_retries,
            )
            try:
                result = build_image(
                    base_image,
                    target_image,
                    custom_tag,
                    target,
                    push,
                    force_build=force_build,
                )
            except Exception as e:
                result = BuildOutput(
                    base_image=base_image,
                    tags=[],
                    error=repr(e),
                    log_path=str(log_path),
                    status="failed",
                )
            remote_check_total += result.remote_check_seconds or 0.0
            build_total += result.build_seconds or 0.0
            result.log_path = str(log_path)
            if result.error:
                logger.error("Build error for %s: %s", base_image, result.error)
                final_result = result
                maybe_reset_buildkit(base_image, target_image, attempt, max_retries)
                if attempt == max_retries - 1:
                    logger.error("Max retries reached for %s. Giving up.", base_image)
                    break
                continue

            # Apply post-build step if provided
            if post_build_fn:
                post_build_started = time.monotonic()
                result = post_build_fn(result, push)
                post_build_total += time.monotonic() - post_build_started
                result.log_path = str(log_path)
                if result.error:
                    result.status = "failed"
                    logger.error(
                        "Post-build error for %s: %s", base_image, result.error
                    )
                    final_result = result
                    maybe_reset_buildkit(base_image, target_image, attempt, max_retries)
                    if attempt == max_retries - 1:
                        logger.error(
                            "Max retries reached for %s. Giving up.", base_image
                        )
                        break
                    continue

            final_result = result
            break

    if final_result is None:
        raise RuntimeError("Unreachable code reached in _build_with_logging")

    final_result.attempt_count = attempts_used
    final_result.started_at = overall_started_at
    final_result.finished_at = _utcnow_iso()
    final_result.duration_seconds = _round_duration(
        time.monotonic() - overall_started_monotonic
    )
    final_result.remote_check_seconds = _round_duration(remote_check_total)
    final_result.build_seconds = _round_duration(build_total)
    final_result.post_build_seconds = _round_duration(post_build_total)
    if final_result.error:
        final_result.status = "failed"

    return final_result


def _update_pbar(
    pbar: tqdm,
    built: int,
    skipped: int,
    failures: int,
    running: int,
    sample: str | None,
    last_event: str | None,
):
    postfix = f"🛠 {built}  ⏭ {skipped}  ❌ {failures}  🏃 {running}"
    if sample:
        postfix += f" ({sample})"
    if last_event:
        pbar.set_description(last_event)
    pbar.set_postfix_str(postfix, refresh=True)


def default_build_output_dir(
    dataset: str, split: str, base_dir: Path | None = None
) -> Path:
    """
    Default: ./builds/<dataset>/<split>
    Keeps build outputs in one predictable place, easy to .gitignore.
    """
    root = (base_dir or Path.cwd()) / "builds" / dataset / split
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_all_images(
    base_images: list[str],
    target: TargetType,
    build_dir: Path,
    image: str = EVAL_AGENT_SERVER_IMAGE,
    push: bool = False,
    base_image_to_custom_tag_fn: Callable[[str], str] | None = None,
    max_workers: int = 1,
    dry_run: bool = False,
    force_build: bool = False,
    max_retries: int = 3,
    post_build_fn: Callable[[BuildOutput, bool], BuildOutput] | None = None,
) -> int:
    """
    Build all specified base images concurrently, logging output and
    writing a manifest file. Each build is automatically retried on failure.

    Args:
        base_images: List of base images to build from.
        target: Build target type.
        build_dir: Directory to store build logs and manifest.
        image: Target image name for built images.
        push: Whether to push images via buildx.
        base_image_to_custom_tag_fn: Function to extract a custom tag from a base image.
            Evaluated before scheduling builds so it can safely be a closure.
        max_workers: Number of concurrent builds.
        dry_run: If True, only list base images without building.
        force_build: If True, rebuild even when matching remote images already exist.
        max_retries: Number of times to retry each failed build (default: 3).
        post_build_fn: Optional callback called after each successful build.
            Receives (build_result, push) and returns modified BuildOutput.
            If it returns an error, the build is retried.

    Returns:
        Exit code: 0 if all builds succeeded, 1 if any failed.
    """

    build_log_dir = build_dir / "logs"
    manifest_file = build_dir / "manifest.jsonl"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("\n".join(base_images))
        return 0

    built = 0
    skipped = 0
    failures = 0
    mu = Lock()
    results: list[BuildOutput] = []
    overall_started_monotonic = time.monotonic()

    # Batch/prune settings (tunable via env to control disk usage on sticky runners)
    # Default to smaller batches and more aggressive pruning on shared runners.
    batch_size = int(os.getenv("BUILD_BATCH_SIZE", "15"))
    prune_keep_storage_gb = int(os.getenv("BUILDKIT_PRUNE_KEEP_GB", "60"))
    prune_threshold_pct = float(os.getenv("BUILDKIT_PRUNE_THRESHOLD_PCT", "60"))
    # Prune aggressively by default; filters like "unused-for=12h" prevented GC from
    # reclaiming layers created during the current run, leading to disk exhaustion.
    prune_filters: list[str] | None = None

    def _chunks(seq: list[str], size: int):
        if size <= 0:
            yield seq
            return
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    batches = list(_chunks(base_images, batch_size or len(base_images)))
    total_batches = len(batches)

    with (
        manifest_file.open("w") as writer,
        tqdm(
            total=len(base_images), desc="Building agent-server images", leave=True
        ) as pbar,
    ):
        _update_pbar(pbar, built, skipped, failures, 0, None, "Queueing")

        for batch_idx, batch in enumerate(batches, start=1):
            if not batch:
                continue

            batch_started_monotonic = time.monotonic()
            logger.info(
                "Starting batch %d/%d (%d images)", batch_idx, total_batches, len(batch)
            )
            in_progress: set[str] = set()
            batch_built = 0
            batch_skipped = 0
            batch_failures = 0

            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {}
                for base in batch:
                    in_progress.add(base)
                    resolved_tag = (
                        base_image_to_custom_tag_fn(base)
                        if base_image_to_custom_tag_fn
                        else ""
                    )
                    fut = ex.submit(
                        _build_with_logging,
                        log_dir=build_log_dir,
                        base_image=base,
                        target_image=image,
                        custom_tag=resolved_tag,
                        target=target,
                        push=push,
                        force_build=force_build,
                        max_retries=max_retries,
                        post_build_fn=post_build_fn,
                    )
                    futures[fut] = base

                _update_pbar(
                    pbar,
                    built,
                    skipped,
                    failures,
                    len(in_progress),
                    next(iter(in_progress), None),
                    f"Batch {batch_idx}/{total_batches} running",
                )

                for fut in as_completed(futures):
                    base = futures[fut]
                    status = None
                    try:
                        result: BuildOutput = fut.result()
                    except Exception as e:
                        logger.error("Build failed for %s: %r", base, e)
                        result = BuildOutput(
                            base_image=base,
                            tags=[],
                            error=repr(e),
                            status="failed",
                        )

                    writer.write(result.model_dump_json() + "\n")
                    writer.flush()
                    results.append(result)

                    with mu:
                        if result.error or not result.tags:
                            failures += 1
                            batch_failures += 1
                            status = "❌ Failed"
                        elif result.status == "skipped_remote_exists":
                            skipped += 1
                            batch_skipped += 1
                            status = "⏭ Skipped"
                        else:
                            built += 1
                            batch_built += 1
                            status = "✅ Built"

                    in_progress.discard(base)
                    pbar.update(1)
                    _update_pbar(
                        pbar,
                        built,
                        skipped,
                        failures,
                        len(in_progress),
                        next(iter(in_progress), None),
                        status,
                    )
                    logger.debug(
                        "Image %s completed status=%s attempts=%d duration=%ss build=%ss remote_check=%ss post_build=%ss",
                        base,
                        result.status,
                        result.attempt_count,
                        result.duration_seconds,
                        result.build_seconds,
                        result.remote_check_seconds,
                        result.post_build_seconds,
                    )

            used, total = buildkit_disk_usage()
            if total > 0:
                logger.info(
                    "BuildKit usage after batch %d/%d: %.2f%% (%0.2f GiB / %0.2f GiB)",
                    batch_idx,
                    total_batches,
                    (used / total) * 100,
                    used / (1 << 30),
                    total / (1 << 30),
                )

            if prune_keep_storage_gb and prune_keep_storage_gb > 0:
                pruned = maybe_prune_buildkit_cache(
                    keep_storage_gb=prune_keep_storage_gb,
                    threshold_pct=prune_threshold_pct,
                    filters=prune_filters,
                )
                if pruned:
                    logger.info(
                        "Pruned BuildKit cache after batch %d/%d (keep=%d GiB, threshold=%.1f%%)",
                        batch_idx,
                        total_batches,
                        prune_keep_storage_gb,
                        prune_threshold_pct,
                    )
                else:
                    logger.info(
                        "No prune needed after batch %d/%d (threshold %.1f%%)",
                        batch_idx,
                        total_batches,
                        prune_threshold_pct,
                    )
            batch_duration = time.monotonic() - batch_started_monotonic
            batch_throughput = (
                (len(batch) / batch_duration) * 3600 if batch_duration else 0.0
            )
            logger.info(
                "Finished batch %d/%d in %.1fs: built=%d skipped=%d failed=%d throughput=%.1f images/hour",
                batch_idx,
                total_batches,
                batch_duration,
                batch_built,
                batch_skipped,
                batch_failures,
                batch_throughput,
            )

    summary_file = build_dir / "build-summary.json"
    summary = summarize_build_records(
        [result.model_dump(mode="json") for result in results],
        manifest_files=1 if results else 0,
    )
    summary_file.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    overall_duration = time.monotonic() - overall_started_monotonic
    throughput = (
        (len(base_images) / overall_duration) * 3600 if overall_duration else 0.0
    )
    logger.info(
        "Done in %.1fs. Built=%d Skipped=%d Failed=%d Retried=%d Throughput=%.1f images/hour Manifest=%s Summary=%s",
        overall_duration,
        built,
        skipped,
        failures,
        summary.retried,
        throughput,
        str(manifest_file),
        str(summary_file),
    )
    return 1 if failures else 0
