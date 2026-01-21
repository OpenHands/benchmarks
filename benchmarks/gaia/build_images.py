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
from pathlib import Path

from benchmarks.gaia.constants import GAIA_BASE_IMAGE, GAIA_DEFAULT_SPLIT
from benchmarks.utils.build_utils import (
    BuildOutput,
    _get_sdk_submodule_info,
    build_all_images,
    default_build_output_dir,
    get_build_parser,
    run_docker_build_layer,
)
from openhands.sdk import get_logger


logger = get_logger(__name__)

# MCP layer Dockerfile
MCP_DOCKERFILE = Path(__file__).with_name("Dockerfile.gaia")


def build_gaia_mcp_layer(base_gaia_image: str, push: bool = False) -> BuildOutput:
    """
    Build the GAIA image with MCP server pre-installed, overriding the base image.

    Args:
        base_gaia_image: The base GAIA image (e.g., ghcr.io/openhands/eval-agent-server:SHA-gaia)
        push: If True, push to registry. If False, load locally.

    Returns:
        BuildOutput with the same image tag or error.
    """
    logger.info(
        "Building MCP-enhanced GAIA image (overriding base): %s", base_gaia_image
    )

    return run_docker_build_layer(
        dockerfile=MCP_DOCKERFILE,
        context=MCP_DOCKERFILE.parent.parent.parent,  # Root of benchmarks repo
        tags=[base_gaia_image],
        build_args={"SDK_IMAGE": base_gaia_image},
        push=push,
        platform="linux/amd64",
        load=not push,
    )


def main(argv: list[str]) -> int:
    parser = get_build_parser()
    args = parser.parse_args(argv)

    # GAIA only needs one universal image for all instances
    base_images = [GAIA_BASE_IMAGE]

    logger.info(f"Building GAIA agent server image from base: {GAIA_BASE_IMAGE}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Image: {args.image}")
    logger.info(f"Push: {args.push}")

    def tag_fn(_base: str) -> str:
        return f"gaia-{args.target}"

    # Build base GAIA image
    build_dir = default_build_output_dir("gaia", GAIA_DEFAULT_SPLIT)
    exit_code = build_all_images(
        base_images=base_images,
        target=args.target,
        build_dir=build_dir,
        image=args.image,
        push=args.push,
        max_workers=1,  # Only building one image
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        base_image_to_custom_tag_fn=tag_fn,
    )

    if exit_code != 0:
        logger.error("Base GAIA image build failed")
        return exit_code

    # Build MCP-enhanced layer after base image succeeds
    git_ref, git_sha, sdk_version = _get_sdk_submodule_info()
    base_gaia_image = f"{args.image}:{git_sha[:7]}-gaia-{args.target}"

    logger.info("Building MCP-enhanced GAIA image from base: %s", base_gaia_image)
    mcp_result = build_gaia_mcp_layer(base_gaia_image, push=args.push)

    if mcp_result.error:
        logger.error("MCP layer build failed: %s", mcp_result.error)
        return 1

    logger.info(
        "Successfully built MCP-enhanced GAIA image: %s",
        mcp_result.tags[0] if mcp_result.tags else "unknown",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
