#!/usr/bin/env python3
"""
Build a universal agent-server image for GAIA benchmark.

Unlike SWE-bench which requires per-instance images with specific repository environments,
GAIA uses a single universal image for all instances since they share the same Python+Node.js environment.

Example:
  uv run benchmarks/gaia/build_images.py \
    --image ghcr.io/openhands/eval-agent-server --target binary-minimal --push
"""

import sys

from benchmarks.utils.build_utils import (
    _get_sdk_submodule_info,
    build_all_images,
    default_build_output_dir,
    get_build_parser,
)
from benchmarks.utils.image_utils import image_exists
from openhands.agent_server.docker.build import BuildOptions
from openhands.sdk import get_logger


logger = get_logger(__name__)

# GAIA base image: Python 3.12 + Node.js 22 (default for agent server)
GAIA_BASE_IMAGE = "nikolaik/python-nodejs:python3.12-nodejs22"


def gaia_tag_fn(_: str) -> str:
    """Return custom tag for GAIA images (all use 'gaia' tag)."""
    return "gaia"


def main(argv: list[str]) -> int:
    parser = get_build_parser()
    args = parser.parse_args(argv)

    # GAIA only needs one universal image for all instances
    base_images = [GAIA_BASE_IMAGE]

    logger.info(f"Building GAIA agent server image from base: {GAIA_BASE_IMAGE}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Image: {args.image}")
    logger.info(f"Push: {args.push}")

    # Skip build if expected tags already exist in the registry
    git_ref, git_sha, sdk_version = _get_sdk_submodule_info()
    opts = BuildOptions(
        base_image=GAIA_BASE_IMAGE,
        custom_tags=gaia_tag_fn(GAIA_BASE_IMAGE),
        image=args.image,
        target=args.target,
        platforms=["linux/amd64"],
        push=args.push,
        git_ref=git_ref,
        git_sha=git_sha,
        sdk_version=sdk_version,
    )
    expected_tags = opts.all_tags[0]
    if expected_tags and all(image_exists(tag) for tag in expected_tags):
        logger.info(
            "All GAIA images already exist: %s. Skipping build.",
            ", ".join(expected_tags),
        )
        return 0

    build_dir = default_build_output_dir("gaia", "validation")
    return build_all_images(
        base_images=base_images,
        target=args.target,
        build_dir=build_dir,
        image=args.image,
        push=args.push,
        max_workers=1,  # Only building one image
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        base_image_to_custom_tag_fn=gaia_tag_fn,  # Tag all with "gaia"
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
