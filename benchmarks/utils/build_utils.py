#!/usr/bin/env python3
"""
Shared utilities for batch building agent-server images.
"""

import argparse
import contextlib
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Callable

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from openhands.agent_server.docker.build import BuildOptions, TargetType, build
from openhands.sdk import get_logger


logger = get_logger(__name__)


class BuildOutput(BaseModel):
    time: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    base_image: str
    tags: list[str]
    error: str | None = None
    log_path: str | None = None


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
    return parser


def build_image(
    base_image: str,
    target_image: str,
    custom_tag: str,
    target: TargetType = "source-minimal",
    push: bool = False,
) -> BuildOutput:
    opts = BuildOptions(
        base_image=base_image,
        custom_tags=custom_tag,
        image=target_image,
        target=target,
        # SWE-Bench only supports linux/amd64 images
        platforms=["linux/amd64"],
        push=push,
    )
    tags = build(opts)
    return BuildOutput(base_image=base_image, tags=tags, error=None)


def _build_with_logging(
    log_dir: Path,
    base_image: str,
    target_image: str,
    target: TargetType = "source-minimal",
    push: bool = False,
    base_image_to_custom_tag_fn: Callable[[str], str] | None = None,
    max_retries: int = 3,
) -> BuildOutput:
    """
    Module-level function for building a single image with output capture.
    Must be at module level to be picklable for ProcessPoolExecutor.
    Automatically retries failed builds up to max_retries times.
    """
    custom_tag = ""
    if base_image_to_custom_tag_fn:
        custom_tag = base_image_to_custom_tag_fn(base_image)

    assert max_retries >= 1, "max_retries must be at least 1"
    for attempt in range(max_retries):
        with capture_output(base_image, log_dir) as log_path:
            if attempt > 0:
                logger.info(
                    f"Retrying build for {base_image} (attempt {attempt + 1}/{max_retries})"
                )
            result = build_image(base_image, target_image, custom_tag, target, push)
            result.log_path = str(log_path)
            if not result.error:
                return result
        logger.error("Build error for %s: %s", base_image, result.error)
        if attempt == max_retries - 1:
            logger.error("Max retries reached for %s. Giving up.", base_image)
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
        base_image_to_custom_tag_fn: Function to extract custom tag from base image.
        max_workers: Number of concurrent builds.
        dry_run: If True, only list base images without building.
        max_retries: Number of times to retry each failed build (default: 2).

    Returns:
        Exit code: 0 if all builds succeeded, 1 if any failed.
    """

    build_log_dir = build_dir / "logs"
    manifest_path = build_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("\n".join(base_images))
        return 0

    successes = 0
    failures = 0
    in_progress: set[str] = set()
    mu = Lock()

    with (
        manifest_path.open("w") as writer,
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
                fut = ex.submit(
                    _build_with_logging,
                    build_log_dir,
                    base,
                    image,
                    target,
                    push,
                    base_image_to_custom_tag_fn,
                    max_retries,
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
        str(manifest_path),
    )
    return 1 if failures else 0
