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

from benchmarks.utils.build_utils import (
    BuildOutput,
    _get_sdk_submodule_info,
    build_all_images,
    default_build_output_dir,
    get_build_parser,
    run_docker_build_layer,
)
from benchmarks.utils.image_utils import image_exists
from openhands.agent_server.docker.build import BuildOptions
from openhands.sdk import get_logger


logger = get_logger(__name__)

# GAIA base image: Python 3.12 + Node.js 22 (default for agent server)
GAIA_BASE_IMAGE = "nikolaik/python-nodejs:python3.12-nodejs22"
# MCP layer Dockerfile
MCP_DOCKERFILE = Path(__file__).with_name("Dockerfile.gaia")


def gaia_tag_fn(_: str) -> str:
    """Return custom tag for GAIA images (all use 'gaia' tag)."""
    return "gaia"


def build_gaia_mcp_layer(base_gaia_image: str, push: bool = False) -> BuildOutput:
    """
    Build the GAIA image with MCP server pre-installed.

    Args:
        base_gaia_image: The base GAIA image (e.g., ghcr.io/openhands/eval-agent-server:SHA-gaia-binary)
        push: If True, push to registry. If False, load locally.

    Returns:
        BuildOutput with the MCP-enhanced image tag or error.
    """
    # Compute MCP image tag by appending -with-mcp to the base tag
    if ":" in base_gaia_image:
        repo, tag = base_gaia_image.rsplit(":", 1)
        mcp_image = f"{repo}:{tag}-with-mcp"
    else:
        mcp_image = f"{base_gaia_image}-with-mcp"

    # Check if MCP image already exists when pushing
    if push and image_exists(mcp_image):
        logger.info("MCP-enhanced GAIA image already exists: %s. Skipping build.", mcp_image)
        return BuildOutput(base_image=base_gaia_image, tags=[mcp_image], error=None)

    # Check if base image exists when pushing
    if push and not image_exists(base_gaia_image):
        return BuildOutput(
            base_image=base_gaia_image,
            tags=[],
            error=(
                f"Base GAIA image {base_gaia_image} not found in registry. "
                "Build and push it before adding MCP layer."
            ),
        )

    if not MCP_DOCKERFILE.exists():
        return BuildOutput(
            base_image=base_gaia_image,
            tags=[],
            error=f"MCP Dockerfile not found at {MCP_DOCKERFILE}",
        )

    logger.info("Building MCP-enhanced GAIA image: %s", mcp_image)
    logger.info("  Base image: %s", base_gaia_image)
    logger.info("  MCP image:  %s", mcp_image)

    return run_docker_build_layer(
        dockerfile=MCP_DOCKERFILE,
        context=MCP_DOCKERFILE.parent.parent.parent,  # Root of benchmarks repo
        tags=[mcp_image],
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
    exit_code = build_all_images(
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

    if exit_code != 0:
        logger.error("Base GAIA image build failed")
        return exit_code

    # Build MCP-enhanced layer after base image succeeds
    # Determine the base GAIA image tag (from the expected tags computed above)
    if expected_tags:
        base_gaia_image = expected_tags[0]  # Use the first tag
    else:
        # Fallback: construct tag manually if image check was skipped
        git_ref, git_sha, sdk_version = _get_sdk_submodule_info()
        base_gaia_image = f"{args.image}:{git_sha[:7]}-gaia-{args.target}"

    logger.info("Building MCP-enhanced GAIA image from base: %s", base_gaia_image)
    mcp_result = build_gaia_mcp_layer(base_gaia_image, push=args.push)

    if mcp_result.error:
        logger.error("MCP layer build failed: %s", mcp_result.error)
        return 1

    logger.info("Successfully built MCP-enhanced GAIA image: %s", mcp_result.tags[0] if mcp_result.tags else "unknown")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
