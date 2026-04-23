#!/usr/bin/env python3
"""
Build agent-server images for Commit0 repositories.

Example:
  uv run benchmarks/commit0/build_images.py \
    --dataset wentingzhao/commit0_combined --split test --repo-split lite \
    --image ghcr.io/openhands/eval-agent-server --push --max-workers 16
"""

import hashlib
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import cast

from commit0.harness.constants import SPLIT
from tqdm.auto import tqdm

from benchmarks.commit0.config import BUILD_DEFAULTS, INFER_DEFAULTS
from benchmarks.swebench.build_base_images import (
    _get_sdk_submodule_info,
    build_builder_image,
)
from benchmarks.swebench.constants import TargetType
from benchmarks.utils.build_utils import (
    BuildOutput,
    _update_pbar,
    build_all_images,
    capture_output,
    default_build_output_dir,
    get_build_parser,
)
from benchmarks.utils.image_utils import remote_image_exists
from benchmarks.utils.version import IMAGE_TAG_PREFIX
from openhands.sdk import get_logger


logger = get_logger(__name__)
DEFAULT_DOCKER_IMAGE_PREFIX = "docker.io/wentingzhao/"
SOURCE_TARGETS = {"source", "source-minimal"}

COMMIT0_AGENT_LAYER_DOCKERFILE = (
    Path(__file__).resolve().parent.parent / "utils" / "Dockerfile.agent-layer-commit0"
)


def get_base_docker_image(
    repo_name: str,
    docker_image_prefix: str | None = None,
) -> str:
    """Get the upstream Commit0 base image for a repository."""
    prefix = docker_image_prefix or os.getenv(
        "EVAL_DOCKER_IMAGE_PREFIX", DEFAULT_DOCKER_IMAGE_PREFIX
    )
    return (prefix.rstrip("/") + "/" + repo_name).lower() + ":v0"


def extract_custom_tag(base_image: str) -> str:
    """Extract Commit0 custom tag from a base image name."""
    repo_tag = base_image.rsplit("/", 1)[-1]
    repo_name = repo_tag.split(":", 1)[0].lower()
    return f"commit0-{repo_name}"


def get_agent_server_image_tag(
    base_image: str,
    target: str,
    image: str,
) -> str:
    """Build the final agent-server image tag used by commit0 run_infer."""
    custom_tag = extract_custom_tag(base_image)
    suffix = f"-{target}" if target != "binary" else ""
    prefix = get_agent_server_image_tag_prefix(target)
    return f"{image}:{prefix}-{custom_tag}{suffix}"


def agent_layer_content_hash() -> str:
    """Return a short hash for the commit0 wrapper Dockerfile contents."""
    content = COMMIT0_AGENT_LAYER_DOCKERFILE.read_text()
    return hashlib.sha256(content.encode()).hexdigest()[:7]


def get_agent_server_image_tag_prefix(target: str) -> str:
    """Return the tag prefix used for commit0 agent images."""
    if target in SOURCE_TARGETS:
        return f"{IMAGE_TAG_PREFIX}-{agent_layer_content_hash()}"
    return IMAGE_TAG_PREFIX


def _load_selected_instances(selected_instances_file: str) -> list[str]:
    selected: list[str] = []
    with open(selected_instances_file, "r", encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if name:
                selected.append(name)
    return selected


def resolve_repos(repo_split: str) -> list[str]:
    """Resolve repository names for a Commit0 repo split."""
    repo_split = repo_split.strip()
    if repo_split in SPLIT:
        repos = list(SPLIT[repo_split])
    else:
        repos = [repo_split]
    return repos


def collect_base_images(
    repo_split: str,
    n_limit: int,
    selected_instances_file: str | None,
    docker_image_prefix: str | None,
) -> list[str]:
    repos = resolve_repos(repo_split)

    if selected_instances_file:
        selected = set(_load_selected_instances(selected_instances_file))
        repos = [repo for repo in repos if repo in selected]

    if n_limit:
        repos = repos[:n_limit]

    if not repos:
        raise ValueError("No Commit0 repositories selected for image build")

    logger.info("Preparing %d Commit0 repos for build", len(repos))
    return [get_base_docker_image(repo, docker_image_prefix) for repo in repos]


def _log_disk(label: str) -> None:
    """Log free disk space to stderr so it appears in the GH Actions log."""
    try:
        usage = shutil.disk_usage("/")
        free_gib = usage.free / (1024**3)
        total_gib = usage.total / (1024**3)
        used_pct = (usage.used / usage.total) * 100
        msg = f"[disk] {label}: {free_gib:.1f} GiB free / {total_gib:.1f} GiB total ({used_pct:.1f}% used)\n"
        os.write(2, msg.encode())  # write to original fd 2 (bypasses capture_output redirect)
    except Exception:
        pass


def _assemble_commit0_image(
    *,
    base_image: str,
    builder_tag: str,
    final_tag: str,
    git_sha: str,
    push: bool,
) -> BuildOutput:
    cmd = [
        "docker",
        "buildx",
        "build",
        "--file",
        str(COMMIT0_AGENT_LAYER_DOCKERFILE),
        "--build-arg",
        f"BASE_IMAGE={base_image}",
        "--build-arg",
        f"BUILDER_IMAGE={builder_tag}",
        "--build-arg",
        f"OPENHANDS_BUILD_GIT_SHA={git_sha}",
        "--tag",
        final_tag,
        "--platform",
        "linux/amd64",
    ]
    if push:
        cmd.append("--push")
    else:
        cmd.append("--load")
    cmd.append(str(COMMIT0_AGENT_LAYER_DOCKERFILE.parent))

    header = f"\n=== commit0 docker build: {base_image} → {final_tag} ===\n"
    os.write(2, header.encode())
    _log_disk(f"pre-build {base_image}")

    # Run without capture_output so workers stream directly to the inherited OS
    # file descriptors (fd 1/2 = GH Actions log). This makes output visible in
    # the step log in real-time — critical for diagnosing mid-build failures
    # where the runner is killed before any Python-level logging can flush.
    proc = subprocess.run(cmd)

    _log_disk(f"post-build {base_image}")
    footer = f"=== done: {base_image} exit={proc.returncode} ===\n"
    os.write(2, footer.encode())

    if proc.returncode != 0:
        return BuildOutput(
            base_image=base_image,
            tags=[],
            error=f"docker buildx build exited {proc.returncode} for {base_image}",
        )

    return BuildOutput(base_image=base_image, tags=[final_tag], error=None)


def _assemble_commit0_with_logging(
    *,
    log_dir: Path,
    base_image: str,
    builder_tag: str,
    final_tag: str,
    git_sha: str,
    push: bool,
    max_retries: int,
    force_build: bool,
) -> BuildOutput:
    import time

    if not force_build and remote_image_exists(final_tag):
        logger.info("Agent image %s already exists. Skipping.", final_tag)
        return BuildOutput(
            base_image=base_image,
            tags=[final_tag],
            error=None,
            status="skipped_remote_exists",
            skip_reason="remote_image_exists",
        )

    assert max_retries >= 1
    for attempt in range(max_retries):
        with capture_output(base_image, log_dir) as log_path:
            if attempt > 0:
                logger.info(
                    "Retrying commit0 assembly for %s (attempt %d/%d)",
                    base_image,
                    attempt + 1,
                    max_retries,
                )
                retry_delay = float(os.getenv("BUILD_RETRY_DELAY_SEC", "2"))
                time.sleep(retry_delay * (1 + attempt))
            try:
                result = _assemble_commit0_image(
                    base_image=base_image,
                    builder_tag=builder_tag,
                    final_tag=final_tag,
                    git_sha=git_sha,
                    push=push,
                )
            except Exception as e:
                result = BuildOutput(
                    base_image=base_image,
                    tags=[],
                    error=repr(e),
                )
            result.log_path = str(log_path)
            if result.error:
                logger.error(
                    "Commit0 assembly error for %s: %s", base_image, result.error
                )
                if attempt == max_retries - 1:
                    return result
                continue
            return result

    raise RuntimeError("Unreachable")


def assemble_commit0_agent_images(
    *,
    base_images: list[str],
    builder_tag: str,
    build_dir: Path,
    target_image: str,
    target: str,
    push: bool,
    max_workers: int,
    max_retries: int,
    force_build: bool,
) -> int:
    """Assemble commit0 source images directly from the real upstream base images."""
    _, git_sha, _ = _get_sdk_submodule_info()
    build_log_dir = build_dir / "assembly-logs"
    manifest_file = build_dir / "manifest.jsonl"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    _log_disk("pre-assembly")
    logger.info(
        "Starting commit0 assembly: %d images, max_workers=%d", len(base_images), max_workers
    )

    built = 0
    skipped = 0
    failures = 0
    mu = Lock()

    # Process in batches so we can prune BuildKit cache from the main process
    # between batches without racing against active worker builds.
    # ubuntu-24.04 runners have ~14 GiB free; without inter-batch pruning the
    # BuildKit content store fills up and the runner is OOM/disk-killed.
    batch_size = max_workers
    batches = [base_images[i : i + batch_size] for i in range(0, len(base_images), batch_size)]
    keep_gb = int(os.getenv("OPENHANDS_BUILDKIT_KEEP_STORAGE_GB", "8"))

    with (
        manifest_file.open("w") as writer,
        tqdm(
            total=len(base_images), desc="Assembling commit0 agent images", leave=True
        ) as pbar,
    ):
        _update_pbar(pbar, built, skipped, failures, 0, None, "Queueing")

        for batch_idx, batch in enumerate(batches, start=1):
            logger.info(
                "Batch %d/%d (%d images)", batch_idx, len(batches), len(batch)
            )
            _log_disk(f"batch {batch_idx} start")

            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {}
                in_progress: set[str] = set()
                for base_image in batch:
                    in_progress.add(base_image)
                    final_tag = get_agent_server_image_tag(base_image, target, target_image)
                    fut = ex.submit(
                        _assemble_commit0_with_logging,
                        log_dir=build_log_dir,
                        base_image=base_image,
                        builder_tag=builder_tag,
                        final_tag=final_tag,
                        git_sha=git_sha,
                        push=push,
                        max_retries=max_retries,
                        force_build=force_build,
                    )
                    futures[fut] = base_image

                _update_pbar(
                    pbar,
                    built,
                    skipped,
                    failures,
                    len(in_progress),
                    next(iter(in_progress), None),
                    f"Batch {batch_idx}/{len(batches)}",
                )

                for fut in as_completed(futures):
                    base_image = futures[fut]
                    try:
                        result: BuildOutput = fut.result()
                    except Exception as e:
                        logger.error("Commit0 assembly failed for %s: %r", base_image, e)
                        result = BuildOutput(base_image=base_image, tags=[], error=repr(e))

                    writer.write(result.model_dump_json() + "\n")
                    writer.flush()

                    with mu:
                        if result.error or not result.tags:
                            failures += 1
                            status = "❌ Failed"
                            logger.error(
                                "Assembly FAILED for %s: %s", base_image, result.error
                            )
                        elif result.status == "skipped_remote_exists":
                            skipped += 1
                            status = "⏭ Skipped"
                        else:
                            built += 1
                            status = "✅ Done"

                    in_progress.discard(base_image)
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

            # All workers in this batch are done — safe to prune from main process.
            _log_disk(f"batch {batch_idx} end (pre-prune)")
            subprocess.run(
                ["docker", "system", "prune", "-f"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["docker", "builder", "prune", "-af", "--keep-storage", f"{keep_gb}g"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _log_disk(f"batch {batch_idx} end (post-prune)")

    logger.info(
        "Commit0 image assembly done. Built=%d Skipped=%d Failed=%d Manifest=%s",
        built,
        skipped,
        failures,
        str(manifest_file),
    )
    return 1 if failures else 0


def build_commit0_images(
    *,
    base_images: list[str],
    target: TargetType,
    build_dir: Path,
    image: str,
    push: bool,
    max_workers: int,
    build_batch_size: int | None,
    dry_run: bool,
    force_build: bool,
    max_retries: int,
) -> int:
    if target in SOURCE_TARGETS:
        if dry_run:
            print("\n".join(base_images))
            return 0

        logger.info("Using phased source-image assembly for commit0 target %s", target)
        builder_result = build_builder_image(
            push=push,
            platform="linux/amd64",
            force_build=force_build,
        )
        if builder_result.error or not builder_result.tags:
            logger.error("Failed to build shared SDK builder image: %s", builder_result)
            return 1

        return assemble_commit0_agent_images(
            base_images=base_images,
            builder_tag=builder_result.tags[0],
            build_dir=build_dir,
            target_image=image,
            target=target,
            push=push,
            max_workers=max_workers,
            max_retries=max_retries,
            force_build=force_build,
        )

    return build_all_images(
        base_images=base_images,
        target=target,
        build_dir=build_dir,
        image=image,
        push=push,
        max_workers=max_workers,
        build_batch_size=build_batch_size,
        dry_run=dry_run,
        force_build=force_build,
        max_retries=max_retries,
        base_image_to_custom_tag_fn=extract_custom_tag,
    )


def main(argv: list[str]) -> int:
    parser = get_build_parser()
    parser.add_argument(
        "--repo-split",
        type=str,
        help="Commit0 repo split (lite, all, or repo name)",
    )
    parser.add_argument(
        "--docker-image-prefix",
        type=str,
        default="",
        help="Override base image prefix (default: env EVAL_DOCKER_IMAGE_PREFIX)",
    )
    parser.set_defaults(
        dataset=INFER_DEFAULTS["dataset"],
        split=INFER_DEFAULTS["split"],
        repo_split=INFER_DEFAULTS["repo_split"],
        **BUILD_DEFAULTS,
    )
    args = parser.parse_args(argv)

    target = cast(TargetType, args.target)
    docker_image_prefix = args.docker_image_prefix or None

    base_images = collect_base_images(
        repo_split=args.repo_split,
        n_limit=args.n_limit,
        selected_instances_file=args.select,
        docker_image_prefix=docker_image_prefix,
    )

    build_dir = default_build_output_dir(args.dataset, args.split)
    return build_commit0_images(
        base_images=base_images,
        target=target,
        build_dir=build_dir,
        image=args.image,
        push=args.push,
        max_workers=args.max_workers,
        build_batch_size=args.build_batch_size,
        dry_run=args.dry_run,
        force_build=args.force_build,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
