#!/usr/bin/env python3
"""
Helpers for phased benchmark image builds.

This module keeps the base-image build entrypoint and the helper functions used
by the phased path: build the shared builder image, build per-instance base
images, then assemble final images locally.
"""

import argparse
import hashlib
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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
DOCKER_COMMAND_TIMEOUT_SECONDS = max(
    1,
    int(os.getenv("OPENHANDS_DOCKER_COMMAND_TIMEOUT_SECONDS", "2000")),
)
PHASED_BUILD_HEARTBEAT_SECONDS = max(
    1,
    int(os.getenv("OPENHANDS_PHASED_BUILD_HEARTBEAT_SECONDS", "60")),
)
AGENT_LAYER_DOCKERFILE = (
    Path(__file__).parent.parent / "utils" / "Dockerfile.agent-layer"
)


def _get_repo_root() -> Path:
    """Get the repository root using git."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def _get_sdk_dockerfile() -> Path:
    """Locate the SDK Dockerfile from the vendor submodule."""
    benchmarks_root = _get_repo_root()
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


def dockerfile_content_hash() -> str:
    """Return a 7-char SHA-256 hash of the SDK Dockerfile content."""
    content = _get_sdk_dockerfile().read_text()
    return hashlib.sha256(content.encode()).hexdigest()[:7]


def base_image_tag(
    custom_tag: str, image: str = EVAL_BASE_IMAGE, *, content_hash: str
) -> str:
    """Compute the full registry tag for a pre-built base image.

    The tag includes a content hash of the SDK Dockerfile so that
    any change to the Dockerfile automatically invalidates cached images.
    Compute *content_hash* once via ``dockerfile_content_hash()`` and
    pass it to all calls.
    """
    return f"{image}:{content_hash}-{custom_tag}"


def _format_timeout_error(
    cmd: list[str], exc: subprocess.TimeoutExpired, timeout_seconds: int
) -> str:
    output = ""
    if exc.stderr:
        output = exc.stderr.strip()
    elif exc.stdout:
        output = exc.stdout.strip()

    message = f"Command timed out after {timeout_seconds}s: {' '.join(cmd)}"
    if output:
        message += f" | Last output: {output[-500:]}"
    return message


def _run_docker_command(
    cmd: list[str],
    *,
    timeout_seconds: int = DOCKER_COMMAND_TIMEOUT_SECONDS,
) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    """Run a Docker command with timeout handling.

    Returns:
        ``(proc, None)`` when the command completes before the timeout.
        ``(None, error_message)`` when the command times out.
    """
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            print(exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr)
        error = _format_timeout_error(cmd, exc, timeout_seconds)
        logger.error(error)
        return None, error

    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    return proc, None


def _pending_summary(in_progress: set[str], limit: int = 3) -> str:
    pending = sorted(in_progress)
    if not pending:
        return "none"
    sample = ", ".join(pending[:limit])
    if len(pending) > limit:
        sample += f", ... (+{len(pending) - limit} more)"
    return sample


def _yield_completed_futures(
    futures: dict,
    in_progress: set[str],
    phase_name: str,
):
    """Yield completed futures while emitting periodic heartbeat logs.

    Args:
        futures: Mapping of submitted futures to their build identifier.
        in_progress: Caller-maintained set of identifiers still pending.
        phase_name: Label to include in heartbeat log messages.
    """
    pending = set(futures)
    while pending:
        done, pending = wait(
            pending,
            timeout=PHASED_BUILD_HEARTBEAT_SECONDS,
            return_when=FIRST_COMPLETED,
        )
        if not done:
            logger.info(
                "%s still running: pending=%d sample=%s",
                phase_name,
                len(in_progress),
                _pending_summary(in_progress),
            )
            continue
        for fut in done:
            yield fut


def build_base_image(
    base_image: str,
    custom_tag: str,
    image: str = EVAL_BASE_IMAGE,
    push: bool = False,
    platform: str = "linux/amd64",
    force_build: bool = False,
    *,
    content_hash: str,
) -> BuildOutput:
    """Build a single base image using the SDK Dockerfile's base-image-minimal target."""
    dockerfile = _get_sdk_dockerfile()
    tag = base_image_tag(custom_tag, image, content_hash=content_hash)

    # Check registry first
    if not force_build and remote_image_exists(tag):
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
    proc, timeout_error = _run_docker_command(cmd)
    if timeout_error:
        return BuildOutput(base_image=base_image, tags=[], error=timeout_error)
    assert proc is not None

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
    force_build: bool = False,
    *,
    content_hash: str,
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
                result = build_base_image(
                    base_image,
                    custom_tag,
                    image,
                    push,
                    force_build=force_build,
                    content_hash=content_hash,
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
    force_build: bool = False,
) -> int:
    """Build all base images concurrently."""
    build_log_dir = build_dir / "base-logs"
    manifest_file = build_dir / "base-manifest.jsonl"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    content_hash = dockerfile_content_hash()

    if dry_run:
        for base in base_images:
            tag = base_image_tag(
                extract_custom_tag(base), image, content_hash=content_hash
            )
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
                    force_build=force_build,
                    content_hash=content_hash,
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

            for fut in _yield_completed_futures(futures, in_progress, "Base builds"):
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
    force_build: bool = False,
) -> BuildOutput:
    """Build and push the SDK builder image (Phase 0).

    Builds the builder stage from the SDK Dockerfile as a standalone image
    containing /agent-server with the venv. Uses the SDK sdist as build context.
    """
    tag = builder_image_tag(builder_image)

    # BuildOutput.base_image is used as the identifier for this build result.
    # For the builder, we use the builder_image repo name (not a Docker base image).
    build_id = builder_image

    if not force_build and remote_image_exists(tag):
        logger.info("Builder image %s already exists. Skipping.", tag)
        return BuildOutput(base_image=build_id, tags=[tag], error=None)

    logger.info("Building builder image: %s", tag)

    # Builder target needs the SDK source as build context.
    # Use the SDK's _make_build_context to create a clean sdist-based context.
    from openhands.agent_server.docker.build import _make_build_context

    sdk_path = _get_repo_root() / "vendor" / "software-agent-sdk"
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
        proc, timeout_error = _run_docker_command(cmd)
        if timeout_error:
            return BuildOutput(base_image=build_id, tags=[], error=timeout_error)
        assert proc is not None

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
    proc, timeout_error = _run_docker_command(build_cmd)
    build_seconds = round(time.monotonic() - build_started, 3)
    if timeout_error:
        logger.info(
            "[assembly] FAILED %s: build_seconds=%.1f error=%s",
            tag_label,
            build_seconds,
            timeout_error[:200],
        )
        return BuildOutput(base_image=base_tag, tags=[], error=timeout_error)
    assert proc is not None

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
            push_proc, timeout_error = _run_docker_command(push_cmd)
            push_seconds += time.monotonic() - push_started
            if timeout_error:
                failed_pushes.append((t, timeout_error[:200]))
                continue
            assert push_proc is not None

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

    # Issue #495: release disk after a successful assembly. Three tiers:
    #   1. `docker rmi` the just-pushed final tags AND the per-instance base
    #      tag (it's 1:1 with this instance, never reused).
    #   2. `docker image prune -f` turns those now-dangling layers back into
    #      free space in the daemon's overlay2 store.
    #   3. `docker builder prune --keep-storage 40g -f` caps BuildKit's
    #      content-addressed ingest cache (`/var/lib/docker/buildkit/content/
    #      ingest/...`). DOCKER_BUILDKIT=1 makes `docker build` route pulls
    #      through this store, and it is *not* reached by rmi or image prune
    #      — 500+ GB of SWE-bench base-image blobs accumulate here otherwise.
    # All failures are warn-and-continue: a cleanup hiccup should not fail an
    # otherwise-successful assembly.
    if push and pushed_tags:
        rmi_targets = list(dict.fromkeys([*pushed_tags, base_tag]))
        rmi_cmd = ["docker", "rmi", "-f", *rmi_targets]
        rmi_proc, rmi_timeout = _run_docker_command(rmi_cmd)
        if rmi_timeout:
            logger.warning("[assembly] rmi timed out for %s", tag_label)
        elif rmi_proc is not None and rmi_proc.returncode != 0:
            logger.warning(
                "[assembly] rmi failed for %s (rc=%d): %s",
                tag_label,
                rmi_proc.returncode,
                (rmi_proc.stderr or rmi_proc.stdout or "").strip()[:200],
            )

        image_prune_proc, image_prune_timeout = _run_docker_command(
            ["docker", "image", "prune", "-f"]
        )
        if image_prune_timeout:
            logger.warning("[assembly] image prune timed out for %s", tag_label)
        elif image_prune_proc is not None and image_prune_proc.returncode != 0:
            logger.warning(
                "[assembly] image prune failed for %s (rc=%d)",
                tag_label,
                image_prune_proc.returncode,
            )

        # Reported OOM sites in #495 validation runs include
        # /var/lib/docker/tmp (tarsplit staging during image export) and
        # /var/lib/docker/image/overlay2/layerdb/tmp (new-layer commits).
        # `docker system prune -f` sweeps those along with dangling images,
        # stopped containers, and unused networks. Safe alongside active
        # builds: it won't touch tagged images or in-use layers.
        sys_prune_proc, sys_prune_timeout = _run_docker_command(
            ["docker", "system", "prune", "-f"]
        )
        if sys_prune_timeout:
            logger.warning("[assembly] system prune timed out for %s", tag_label)
        elif sys_prune_proc is not None and sys_prune_proc.returncode != 0:
            logger.warning(
                "[assembly] system prune failed for %s (rc=%d)",
                tag_label,
                sys_prune_proc.returncode,
            )

        # Previously defaulted to 40 GiB but that let the cache grow past
        # what the runner could carry alongside 4 concurrent in-flight
        # builds. Drop to 10 GiB; the shared builder image still fits.
        buildkit_cap_gb = int(os.getenv("OPENHANDS_BUILDKIT_KEEP_STORAGE_GB", "10"))
        builder_prune_cmd = [
            "docker",
            "builder",
            "prune",
            "-af",
            "--keep-storage",
            f"{buildkit_cap_gb}g",
        ]
        bp_proc, bp_timeout = _run_docker_command(builder_prune_cmd)
        if bp_timeout:
            logger.warning("[assembly] buildkit prune timed out for %s", tag_label)
        elif bp_proc is not None and bp_proc.returncode != 0:
            logger.warning(
                "[assembly] buildkit prune failed for %s (rc=%d)",
                tag_label,
                bp_proc.returncode,
            )

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
    *,
    content_hash: str,
) -> BuildOutput:
    """Assemble a single agent image with logging and retry."""
    import time

    base_tag = base_image_tag(custom_tag, content_hash=content_hash)
    # Include content_hash so Dockerfile changes invalidate cached assemblies.
    final_tag = f"{target_image}:{sdk_short_sha}-{content_hash}-{custom_tag}-{target}"

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
    content_hash = dockerfile_content_hash()

    # Issue #495: the base-image phase pushes via `buildx --push`, which still
    # routes every blob through BuildKit's content-addressed ingest cache. On
    # SWT-bench with 433 fat SWE-bench base images, this leaves 100+ GB of
    # cache before assembly even starts. Reset the cache fully here so the
    # per-assembly `--keep-storage` cap actually bounds disk usage from this
    # point forward.
    if os.getenv("OPENHANDS_SKIP_PREASSEMBLY_PRUNE") != "1":
        logger.info("Pre-assembly prune of BuildKit cache and dangling images")
        _run_docker_command(
            ["docker", "builder", "prune", "-af", "--keep-storage", "0"]
        )
        _run_docker_command(["docker", "image", "prune", "-f"])

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
                    content_hash=content_hash,
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

            for fut in _yield_completed_futures(
                futures, in_progress, "Agent image assembly"
            ):
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
        default=2,
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
