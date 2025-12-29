#!/usr/bin/env python3
"""
Shared utilities for batch building agent-server images.
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Callable

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.image_utils import image_exists
from openhands.agent_server.docker.build import BuildOptions, TargetType, build
from openhands.sdk import get_logger


logger = get_logger(__name__)


def _read_reset_state(path: Path) -> dict[str, float]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _write_reset_state(path: Path, state: dict[str, float]) -> None:
    try:
        path.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass


def _last_reset_for_kind(kind: str, state: dict[str, float]) -> float:
    if kind == "full":
        return state.get("full", 0.0)
    if kind == "partial":
        return max(state.get("partial", 0.0), state.get("full", 0.0))
    return max(state.values(), default=0.0)


def _should_throttle_reset(
    kind: str, state: dict[str, float], now: float, throttle_sec: int
) -> bool:
    if throttle_sec <= 0:
        return False
    last = _last_reset_for_kind(kind, state)
    return last > 0 and (now - last) < throttle_sec


def _reset_buildkit(reset_kind: str) -> None:
    if os.getenv("BUILDKIT_RESET_ON_FAILURE", "1") == "0":
        return
    if reset_kind not in {"restart", "partial", "full"}:
        return

    lock_path = Path(os.getenv("BUILDKIT_RESET_LOCK", "/tmp/buildkit-reset.lock"))
    state_path = Path(
        os.getenv("BUILDKIT_RESET_STATE", "/tmp/buildkit-reset-state.json")
    )
    throttle_sec = int(os.getenv("BUILDKIT_RESET_THROTTLE_SEC", "300"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    lock_file = lock_path.open("w")
    try:
        try:
            import fcntl

            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except Exception:
            # Best-effort locking; continue without it on unsupported platforms.
            pass

        now = time.time()
        state = _read_reset_state(state_path)
        if _should_throttle_reset(reset_kind, state, now, throttle_sec):
            last = _last_reset_for_kind(reset_kind, state)
            logger.info(
                "Skipping buildx %s reset; last reset %.0fs ago",
                reset_kind,
                now - last,
            )
            return

        if reset_kind == "restart":
            cmds = [["docker", "buildx", "inspect", "--bootstrap"]]
        elif reset_kind == "partial":
            cmds = [
                ["docker", "buildx", "prune", "--force"],
                ["docker", "buildx", "inspect", "--bootstrap"],
            ]
        else:
            cmds = [
                ["docker", "buildx", "prune", "--all", "--force"],
                ["docker", "buildx", "inspect", "--bootstrap"],
            ]

        logger.warning("Resetting buildx (%s) after BuildKit failure", reset_kind)
        for cmd in cmds:
            proc = subprocess.run(cmd, text=True, capture_output=True)
            if proc.stdout:
                logger.info(proc.stdout.strip())
            if proc.stderr:
                logger.warning(proc.stderr.strip())

        state[reset_kind] = now
        _write_reset_state(state_path, state)
    finally:
        try:
            lock_file.close()
        except Exception:
            pass


def _maybe_reset_buildkit(attempt: int, max_retries: int) -> None:
    if attempt >= max_retries - 1:
        return
    if attempt == 0:
        _reset_buildkit("partial")
    else:
        _reset_buildkit("full")


class BuildOutput(BaseModel):
    time: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    base_image: str
    tags: list[str]
    error: str | None = None
    log_path: str | None = None


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
        "--batch-size",
        type=int,
        default=0,
        help="Optional batch size for large runs (use with --batch-index)",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=-1,
        help="Zero-based batch index for large runs (-1 builds all batches)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List base images only, don’t build"
    )
    return parser


def build_image(
    base_image: str,
    target_image: str,
    custom_tag: str,
    target: TargetType = "source-minimal",
    push: bool = False,
) -> BuildOutput:
    # Get SDK info from submodule to ensure tags use the correct SDK SHA
    git_ref, git_sha, sdk_version = _get_sdk_submodule_info()

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
    for t in opts.all_tags:
        # Check if image exists or not
        if image_exists(t):
            logger.info(f"Image {t} already exists. Skipping build.")
            return BuildOutput(base_image=base_image, tags=[t], error=None)
    tags = build(opts)
    return BuildOutput(base_image=base_image, tags=tags, error=None)


def _build_with_logging(
    log_dir: Path,
    base_image: str,
    target_image: str,
    custom_tag: str = "",
    target: TargetType = "source-minimal",
    push: bool = False,
    max_retries: int = 3,
    post_build_fn: Callable[[BuildOutput, bool], BuildOutput] | None = None,
) -> BuildOutput:
    """
    Module-level function for building a single image with output capture.
    Must be at module level to be picklable for ProcessPoolExecutor.
    Automatically retries failed builds up to max_retries times.

    Args:
        custom_tag: Custom tag (already resolved) to pass to build_image.
        post_build_fn: Optional callback called after successful build.
            Receives (build_result, push) and returns modified BuildOutput.
            If it returns an error, the build is retried.
    """
    assert max_retries >= 1, "max_retries must be at least 1"
    for attempt in range(max_retries):
        with capture_output(base_image, log_dir) as log_path:
            if attempt > 0:
                logger.info(
                    f"Retrying build for {base_image} (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(2 + attempt * 2)
            try:
                result = build_image(base_image, target_image, custom_tag, target, push)
            except Exception as e:
                result = BuildOutput(
                    base_image=base_image, tags=[], error=repr(e), log_path=str(log_path)
                )
            result.log_path = str(log_path)
            if result.error:
                logger.error("Build error for %s: %s", base_image, result.error)
                _maybe_reset_buildkit(attempt, max_retries)
                if attempt == max_retries - 1:
                    logger.error("Max retries reached for %s. Giving up.", base_image)
                    return result
                continue

            # Apply post-build step if provided
            if post_build_fn:
                result = post_build_fn(result, push)
                result.log_path = str(log_path)
                if result.error:
                    logger.error(
                        "Post-build error for %s: %s", base_image, result.error
                    )
                    _maybe_reset_buildkit(attempt, max_retries)
                    if attempt == max_retries - 1:
                        logger.error(
                            "Max retries reached for %s. Giving up.", base_image
                        )
                        return result
                    continue

            return result

    raise RuntimeError("Unreachable code reached in _build_with_logging")


def _update_pbar(
    pbar: tqdm,
    successes: int,
    failures: int,
    running: int,
    sample: str | None,
    last_event: str | None,
):
    postfix = f"✅ {successes}  ❌ {failures}  🏃 {running}"
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

    successes = 0
    failures = 0
    in_progress: set[str] = set()
    mu = Lock()

    with (
        manifest_file.open("w") as writer,
        tqdm(
            total=len(base_images), desc="Building agent-server images", leave=True
        ) as pbar,
    ):
        _update_pbar(pbar, successes, failures, 0, None, "Queueing")

        # Single unified path: ProcessPoolExecutor( max_workers = args.max_workers ),
        # even if it's 1. Using processes instead of threads ensures proper isolation
        # of stdout/stderr and logging handlers, preventing output mixing between builds.
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for base in base_images:
                in_progress.add(base)
                # Resolve custom tags before scheduling to avoid pickling issues with closures.
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
                    max_retries=max_retries,
                    post_build_fn=post_build_fn,
                )
                futures[fut] = base

            _update_pbar(
                pbar,
                successes,
                failures,
                len(in_progress),
                next(iter(in_progress), None),
                "Running",
            )

            for fut in as_completed(futures):
                base = futures[fut]
                try:
                    result: BuildOutput = fut.result()
                    writer.write(result.model_dump_json() + "\n")
                    writer.flush()
                    with mu:
                        successes += 1
                    _update_pbar(
                        pbar, successes, failures, len(in_progress), base, "✅ Done"
                    )
                except Exception as e:
                    logger.error("Build failed for %s: %r", base, e)
                    # Write a failure line to manifest; keep going.
                    writer.write(
                        BuildOutput(
                            base_image=base, tags=[], error=repr(e)
                        ).model_dump_json()
                        + "\n"
                    )
                    writer.flush()
                    with mu:
                        failures += 1
                    _update_pbar(
                        pbar, successes, failures, len(in_progress), base, "❌ Failed"
                    )
                finally:
                    with mu:
                        in_progress.discard(base)
                    pbar.update(1)
                    _update_pbar(
                        pbar,
                        successes,
                        failures,
                        len(in_progress),
                        next(iter(in_progress), None),
                        None,
                    )
    logger.info(
        "Done. Built=%d  Failed=%d  Manifest=%s",
        successes,
        failures,
        str(manifest_file),
    )
    return 1 if failures else 0
