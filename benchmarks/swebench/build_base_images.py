#!/usr/bin/env python3
"""
Helpers for phased benchmark image builds.

This module keeps the base-image build entrypoint and the helper functions used
by the phased path: build the shared builder image, build per-instance base
images, then assemble final images locally.
"""

import argparse
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tqdm.auto import tqdm

from benchmarks.swebench.build_images import (
    collect_unique_base_images,
    extract_custom_tag,
)
from benchmarks.utils.build_utils import (
    BuildOutput,
    _get_sdk_submodule_info,
    _update_pbar,
    capture_output,
    default_build_output_dir,
)
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.image_utils import remote_image_exists
from openhands.sdk import get_logger


logger = get_logger(__name__)

# Default registries
EVAL_BASE_IMAGE = os.getenv("OPENHANDS_EVAL_BASE_IMAGE", "ghcr.io/openhands/eval-base")
EVAL_BUILDER_IMAGE = os.getenv(
    "OPENHANDS_EVAL_BUILDER_IMAGE", "ghcr.io/openhands/eval-builder"
)
AGENT_LAYER_DOCKERFILE = (
    Path(__file__).parent.parent / "utils" / "Dockerfile.agent-layer"
)


def _get_sdk_dockerfile() -> Path:
    """Locate the SDK Dockerfile from the vendor submodule.

    NOTE: This assumes the file lives at ``benchmarks/swebench/build_base_images.py``
    so that three ``.parent`` hops reach the repo root.  If this file moves, update
    the traversal or switch to ``git rev-parse --show-toplevel``.
    """
    benchmarks_root = Path(__file__).resolve().parent.parent.parent
    dockerfile = (
        benchmarks_root
        / "vendor"
        / "software-agent-sdk"
        / "openhands-agent-server"
        / "openhands"
        / "agent_server"
        / "docker"
        / "Dockerfile"
    )
    if not dockerfile.exists():
        raise FileNotFoundError(
            f"SDK Dockerfile not found at {dockerfile}. "
            "Make sure submodules are initialized."
        )
    return dockerfile


def base_image_tag(custom_tag: str, image: str = EVAL_BASE_IMAGE) -> str:
    """Compute the full registry tag for a pre-built base image."""
    return f"{image}:{custom_tag}"


def build_base_image(
    base_image: str,
    custom_tag: str,
    image: str = EVAL_BASE_IMAGE,
    push: bool = False,
    platform: str = "linux/amd64",
) -> BuildOutput:
    """Build a single base image using the SDK Dockerfile's base-image-minimal target."""
    dockerfile = _get_sdk_dockerfile()
    tag = base_image_tag(custom_tag, image)

    # Check registry first
    if remote_image_exists(tag):
        logger.info("Base image %s already exists. Skipping.", tag)
        return BuildOutput(base_image=base_image, tags=[tag], error=None)

    # Build with empty context (base-image-minimal doesn't COPY from context)
    cmd = [
        "docker",
        "buildx",
        "build",
        "--file",
        str(dockerfile),
        "--target",
        "base-image-minimal",
        "--build-arg",
        f"BASE_IMAGE={base_image}",
        "--platform",
        platform,
    ]
    cmd.extend(["--tag", tag])

    if push:
        cmd.append("--push")
    else:
        cmd.append("--load")

    # Use the Dockerfile's parent as context (minimal, just needs the Dockerfile)
    cmd.append(str(dockerfile.parent))

    logger.info("Building base image: %s", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True)

    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    if proc.returncode != 0:
        error = (
            proc.stderr.strip()
            or proc.stdout.strip()
            or f"Build failed with exit code {proc.returncode}"
        )
        return BuildOutput(base_image=base_image, tags=[], error=error)

    return BuildOutput(base_image=base_image, tags=[tag], error=None)


def _build_base_with_logging(
    log_dir: Path,
    base_image: str,
    custom_tag: str,
    image: str = EVAL_BASE_IMAGE,
    push: bool = False,
    max_retries: int = 3,
) -> BuildOutput:
    """Build a single base image with logging and retry support."""
    import time

    assert max_retries >= 1
    for attempt in range(max_retries):
        with capture_output(base_image, log_dir) as log_path:
            if attempt > 0:
                logger.info(
                    "Retrying base build for %s (attempt %d/%d)",
                    base_image,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(2 + attempt * 2)
            try:
                result = build_base_image(base_image, custom_tag, image, push)
            except Exception as e:
                result = BuildOutput(
                    base_image=base_image,
                    tags=[],
                    error=repr(e),
                    log_path=str(log_path),
                )
            result.log_path = str(log_path)
            if result.error:
                logger.error("Base build error for %s: %s", base_image, result.error)
                if attempt == max_retries - 1:
                    return result
                continue
            return result

    raise RuntimeError("Unreachable")


def build_all_base_images(
    base_images: list[str],
    build_dir: Path,
    image: str = EVAL_BASE_IMAGE,
    push: bool = False,
    max_workers: int = 1,
    dry_run: bool = False,
    max_retries: int = 3,
) -> int:
    """Build all base images concurrently."""
    build_log_dir = build_dir / "base-logs"
    manifest_file = build_dir / "base-manifest.jsonl"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        for base in base_images:
            tag = base_image_tag(extract_custom_tag(base), image)
            print(f"{base} -> {tag}")
        return 0

    successes = 0
    failures = 0
    mu = Lock()

    with (
        manifest_file.open("w") as writer,
        tqdm(total=len(base_images), desc="Building base images", leave=True) as pbar,
    ):
        _update_pbar(pbar, successes, 0, failures, 0, None, "Queueing")

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            in_progress: set[str] = set()
            for base in base_images:
                in_progress.add(base)
                custom_tag = extract_custom_tag(base)
                fut = ex.submit(
                    _build_base_with_logging,
                    log_dir=build_log_dir,
                    base_image=base,
                    custom_tag=custom_tag,
                    image=image,
                    push=push,
                    max_retries=max_retries,
                )
                futures[fut] = base

            _update_pbar(
                pbar,
                successes,
                0,
                failures,
                len(in_progress),
                next(iter(in_progress), None),
                "Building",
            )

            for fut in as_completed(futures):
                base = futures[fut]
                try:
                    result: BuildOutput = fut.result()
                except Exception as e:
                    logger.error("Base build failed for %s: %r", base, e)
                    result = BuildOutput(base_image=base, tags=[], error=repr(e))

                writer.write(result.model_dump_json() + "\n")
                writer.flush()

                with mu:
                    if result.error or not result.tags:
                        failures += 1
                        status = "❌ Failed"
                    else:
                        successes += 1
                        status = "✅ Done"

                in_progress.discard(base)
                pbar.update(1)
                _update_pbar(
                    pbar,
                    successes,
                    0,
                    failures,
                    len(in_progress),
                    next(iter(in_progress), None),
                    status,
                )

    logger.info(
        "Base images done. Built=%d  Failed=%d  Manifest=%s",
        successes,
        failures,
        str(manifest_file),
    )
    return 1 if failures else 0


def builder_image_tag(builder_image: str = EVAL_BUILDER_IMAGE) -> str:
    """Compute the builder image tag from the SDK SHA."""
    _, git_sha, _ = _get_sdk_submodule_info()
    short_sha = git_sha[:7] if git_sha != "unknown" else "unknown"
    return f"{builder_image}:{short_sha}"


def build_builder_image(
    builder_image: str = EVAL_BUILDER_IMAGE,
    push: bool = False,
    platform: str = "linux/amd64",
) -> BuildOutput:
    """Build and push the SDK builder image (Phase 0).

    Builds the builder stage from the SDK Dockerfile as a standalone image
    containing /agent-server with the venv. Uses the SDK sdist as build context.
    """
    tag = builder_image_tag(builder_image)

    # BuildOutput.base_image is used as the identifier for this build result.
    # For the builder, we use the builder_image repo name (not a Docker base image).
    build_id = builder_image

    if remote_image_exists(tag):
        logger.info("Builder image %s already exists. Skipping.", tag)
        return BuildOutput(base_image=build_id, tags=[tag], error=None)

    logger.info("Building builder image: %s", tag)

    # Builder target needs the SDK source as build context.
    # Use the SDK's _make_build_context to create a clean sdist-based context.
    from openhands.agent_server.docker.build import _make_build_context

    benchmarks_root = Path(__file__).resolve().parent.parent.parent
    sdk_path = benchmarks_root / "vendor" / "software-agent-sdk"
    ctx = _make_build_context(sdk_path)

    try:
        cmd = [
            "docker",
            "buildx",
            "build",
            "--file",
            str(ctx / "Dockerfile"),
            "--target",
            "builder",
            "--platform",
            platform,
            "--tag",
            tag,
        ]
        if push:
            cmd.append("--push")
        else:
            cmd.append("--load")
        cmd.append(str(ctx))

        logger.info("Building builder: %s", " ".join(cmd))
        proc = subprocess.run(cmd, text=True, capture_output=True)

        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)

        if proc.returncode != 0:
            error = (
                proc.stderr.strip()
                or proc.stdout.strip()
                or f"Builder build failed with exit code {proc.returncode}"
            )
            return BuildOutput(base_image=build_id, tags=[], error=error)

        return BuildOutput(base_image=build_id, tags=[tag], error=None)
    finally:
        import shutil

        try:
            shutil.rmtree(ctx)
        except Exception as e:
            logger.warning("Failed to cleanup build context %s: %s", ctx, e)


def assemble_agent_image(
    base_tag: str,
    builder_tag: str,
    final_tags: list[str],
    push: bool = False,
    git_sha: str = "unknown",
) -> BuildOutput:
    """Assemble a final agent image from pre-built base + builder (Phase 2).

    Uses local ``docker build`` + ``docker push`` instead of ``docker buildx
    build --push`` to leverage the local Docker daemon's layer cache across
    many builds in a single job (~455 s/image -> ~70 s/image).

    **Requirement:** A local Docker daemon must be running (``docker info``
    must succeed).  This is satisfied on GitHub Actions runners and most dev
    machines.  Remote-only buildx drivers (e.g. Blacksmith cloud builders)
    will *not* work for this function; use the standard build path instead.
    """
    import time

    if not AGENT_LAYER_DOCKERFILE.exists():
        return BuildOutput(
            base_image=base_tag,
            tags=[],
            error=f"Agent layer Dockerfile not found at {AGENT_LAYER_DOCKERFILE}",
        )

    tag_label = final_tags[0] if final_tags else base_tag
    overall_started = time.monotonic()

    # Step 1: docker build (local daemon, no remote driver)
    build_cmd = [
        "docker",
        "build",
        "--file",
        str(AGENT_LAYER_DOCKERFILE),
        "--build-arg",
        f"BASE_IMAGE={base_tag}",
        "--build-arg",
        f"BUILDER_IMAGE={builder_tag}",
        "--build-arg",
        f"OPENHANDS_BUILD_GIT_SHA={git_sha}",
    ]
    for t in final_tags:
        build_cmd.extend(["--tag", t])
    build_cmd.append(str(AGENT_LAYER_DOCKERFILE.parent))

    logger.info("[assembly] Building: %s", " ".join(build_cmd))
    build_started = time.monotonic()
    proc = subprocess.run(build_cmd, text=True, capture_output=True)
    build_seconds = round(time.monotonic() - build_started, 3)

    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    if proc.returncode != 0:
        error = (
            proc.stderr.strip()
            or proc.stdout.strip()
            or f"docker build failed with exit code {proc.returncode}"
        )
        logger.info(
            "[assembly] FAILED %s: build_seconds=%.1f error=%s",
            tag_label,
            build_seconds,
            error[:200],
        )
        return BuildOutput(base_image=base_tag, tags=[], error=error)

    # Step 2: docker push each tag (collect partial failures)
    push_seconds = 0.0
    pushed_tags: list[str] = []
    failed_pushes: list[tuple[str, str]] = []
    if push:
        for t in final_tags:
            push_cmd = ["docker", "push", t]
            logger.info("[assembly] Pushing: %s", t)
            push_started = time.monotonic()
            push_proc = subprocess.run(push_cmd, text=True, capture_output=True)
            push_seconds += time.monotonic() - push_started

            if push_proc.stdout:
                print(push_proc.stdout, end="")
            if push_proc.stderr:
                print(push_proc.stderr, end="", file=sys.stderr)

            if push_proc.returncode != 0:
                error = (
                    push_proc.stderr.strip()
                    or push_proc.stdout.strip()
                    or f"docker push failed with exit code {push_proc.returncode}"
                )
                failed_pushes.append((t, error[:200]))
            else:
                pushed_tags.append(t)

    if failed_pushes:
        push_seconds = round(push_seconds, 3)
        error_summary = (
            f"Failed to push {len(failed_pushes)}/{len(final_tags)} tags: "
            + "; ".join(f"{t}: {e}" for t, e in failed_pushes)
        )
        logger.info(
            "[assembly] PARTIAL FAIL %s: push_seconds=%.1f pushed=%d/%d error=%s",
            tag_label,
            push_seconds,
            len(pushed_tags),
            len(final_tags),
            error_summary[:300],
        )
        return BuildOutput(base_image=base_tag, tags=pushed_tags, error=error_summary)

    push_seconds = round(push_seconds, 3)
    total_seconds = round(time.monotonic() - overall_started, 3)

    logger.info(
        "[assembly] OK %s: total=%.1fs build=%.1fs push=%.1fs",
        tag_label,
        total_seconds,
        build_seconds,
        push_seconds,
    )

    return BuildOutput(base_image=base_tag, tags=final_tags, error=None)


def _assemble_with_logging(
    log_dir: Path,
    base_image: str,
    custom_tag: str,
    builder_tag: str,
    target_image: str,
    sdk_short_sha: str,
    sdk_full_sha: str,
    target: str,
    push: bool = False,
    max_retries: int = 3,
    force_build: bool = False,
) -> BuildOutput:
    """Assemble a single agent image with logging and retry."""
    import time

    base_tag = base_image_tag(custom_tag)
    # Match the tag format from the SDK's BuildOptions.all_tags
    final_tag = f"{target_image}:{sdk_short_sha}-{custom_tag}-{target}"

    if not force_build and remote_image_exists(final_tag):
        logger.info("Agent image %s already exists. Skipping.", final_tag)
        return BuildOutput(base_image=base_image, tags=[final_tag], error=None)

    assert max_retries >= 1
    for attempt in range(max_retries):
        with capture_output(base_image, log_dir) as log_path:
            if attempt > 0:
                logger.info(
                    "Retrying assembly for %s (attempt %d/%d)",
                    base_image,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(2 + attempt * 2)
            try:
                result = assemble_agent_image(
                    base_tag=base_tag,
                    builder_tag=builder_tag,
                    final_tags=[final_tag],
                    push=push,
                    git_sha=sdk_full_sha,
                )
            except Exception as e:
                result = BuildOutput(
                    base_image=base_image,
                    tags=[],
                    error=repr(e),
                    log_path=str(log_path),
                )
            result.log_path = str(log_path)
            if result.error:
                logger.error("Assembly error for %s: %s", base_image, result.error)
                if attempt == max_retries - 1:
                    return result
                continue

            # Apply wrapping for repos that need docutils/roman (e.g. sphinx-doc)
            from benchmarks.swebench.build_images import (
                should_wrap_custom_tag,
                wrap_image,
            )

            if should_wrap_custom_tag(custom_tag):
                logger.info("Wrapping %s with docutils/roman", final_tag)
                wrap_result = wrap_image(final_tag, push=push)
                if wrap_result.error:
                    result = BuildOutput(
                        base_image=base_image,
                        tags=result.tags,
                        error=f"Wrapping failed: {wrap_result.error}",
                        log_path=str(log_path),
                    )
                    if attempt == max_retries - 1:
                        return result
                    continue

            return result

    raise RuntimeError("Unreachable")


def assemble_all_agent_images(
    base_images: list[str],
    builder_tag: str,
    build_dir: Path,
    target_image: str = EVAL_AGENT_SERVER_IMAGE,
    target: str = "source-minimal",
    push: bool = False,
    max_workers: int = 12,
    max_retries: int = 3,
    force_build: bool = False,
    custom_tag_fn: Callable[[str], str] | None = None,
) -> int:
    """Assemble all agent images using thin Dockerfile (Phase 2)."""
    _, git_sha, _ = _get_sdk_submodule_info()
    sdk_short_sha = git_sha[:7] if git_sha != "unknown" else "unknown"

    build_log_dir = build_dir / "assembly-logs"
    manifest_file = build_dir / "manifest.jsonl"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    built = 0
    skipped = 0
    failures = 0
    mu = Lock()

    with (
        manifest_file.open("w") as writer,
        tqdm(
            total=len(base_images), desc="Assembling agent images", leave=True
        ) as pbar,
    ):
        _update_pbar(pbar, built, skipped, failures, 0, None, "Queueing")

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            in_progress: set[str] = set()
            _tag_fn = custom_tag_fn or extract_custom_tag
            for base in base_images:
                in_progress.add(base)
                custom_tag = _tag_fn(base)
                fut = ex.submit(
                    _assemble_with_logging,
                    log_dir=build_log_dir,
                    base_image=base,
                    custom_tag=custom_tag,
                    builder_tag=builder_tag,
                    target_image=target_image,
                    sdk_short_sha=sdk_short_sha,
                    sdk_full_sha=git_sha,
                    target=target,
                    push=push,
                    max_retries=max_retries,
                    force_build=force_build,
                )
                futures[fut] = base

            _update_pbar(
                pbar,
                built,
                skipped,
                failures,
                len(in_progress),
                next(iter(in_progress), None),
                "Assembling",
            )

            for fut in as_completed(futures):
                base = futures[fut]
                try:
                    result: BuildOutput = fut.result()
                except Exception as e:
                    logger.error("Assembly failed for %s: %r", base, e)
                    result = BuildOutput(base_image=base, tags=[], error=repr(e))

                writer.write(result.model_dump_json() + "\n")
                writer.flush()

                with mu:
                    if result.error or not result.tags:
                        failures += 1
                        status = "❌ Failed"
                    else:
                        built += 1
                        status = "✅ Done"

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

    logger.info(
        "Assembly done. Built=%d  Skipped=%d  Failed=%d  Manifest=%s",
        built,
        skipped,
        failures,
        str(manifest_file),
    )
    return 1 if failures else 0


def get_base_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build pre-built base images for SWE-Bench evaluation."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Verified",
        help="Dataset name",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--image",
        default=EVAL_BASE_IMAGE,
        help="Target repo/name for base images",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push via buildx instead of load locally",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=12,
        help="Concurrent builds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List base images only, don't build",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=0,
        help="Limit number of images (0 = no limit)",
    )
    parser.add_argument(
        "--select",
        type=str,
        default=None,
        help="Path to text file containing instance IDs to select",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per image build",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = get_base_build_parser()
    args = parser.parse_args(argv)

    base_images = collect_unique_base_images(
        args.dataset,
        args.split,
        args.n_limit,
        args.select,
    )
    build_dir = default_build_output_dir(args.dataset, args.split)

    return build_all_base_images(
        base_images=base_images,
        build_dir=build_dir,
        image=args.image,
        push=args.push,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    sys.exit(main())
