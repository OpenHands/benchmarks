#!/usr/bin/env python3
"""
Build pre-built base images for SWE-Bench evaluation.

Base images contain everything from the SWE-bench upstream image through
apt-get/npm setup (the Dockerfile ``base-image-minimal`` stage).  They are
SDK-independent and only need rebuilding when the upstream SWE-bench image
or the Dockerfile's base layers change.

Once base images exist in the registry, the agent-server build
(build_images.py --use-prebuilt-bases) can skip the base-image-minimal
stage entirely, reducing per-image build time from ~154s to ~5-10s.

Example:
  uv run benchmarks/swebench/build_base_images.py \
    --dataset princeton-nlp/SWE-bench_Verified --split test --push
"""

import argparse
import os
import subprocess
import sys
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
    _update_pbar,
    capture_output,
    default_build_output_dir,
)
from benchmarks.utils.image_utils import remote_image_exists
from openhands.sdk import get_logger


logger = get_logger(__name__)

# Default registry for pre-built base images
EVAL_BASE_IMAGE = os.getenv("OPENHANDS_EVAL_BASE_IMAGE", "ghcr.io/openhands/eval-base")


def _get_sdk_dockerfile() -> Path:
    """Locate the SDK Dockerfile from the vendor submodule."""
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
        _update_pbar(pbar, successes, failures, 0, None, "Queueing")

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
