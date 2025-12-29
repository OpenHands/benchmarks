"""Build OpenAgentSafety agent-server image."""

import subprocess
import sys
from pathlib import Path

from benchmarks.utils.build_utils import get_build_parser, run_docker_build_layer
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.image_utils import image_exists
from benchmarks.utils.version import SDK_SHORT_SHA
from openhands.sdk import get_logger


logger = get_logger(__name__)

OPENAGENTSAFETY_BASE_IMAGE = "ghcr.io/sani903/openagentsafety_base_image-image:1.0"
OPENAGENTSAFETY_CUSTOM_TAG = "openagentsafety"
DEFAULT_TARGET = "source-minimal"


def resolve_openagentsafety_image_tag(
    image: str | None = None,
    target: str | None = None,
    sdk_short_sha: str | None = None,
) -> str:
    """Compute the OpenAgentSafety agent-server image tag."""
    image = image or EVAL_AGENT_SERVER_IMAGE
    target = target or DEFAULT_TARGET
    sdk_short_sha = sdk_short_sha or SDK_SHORT_SHA
    tag = f"{image}:{sdk_short_sha}-{OPENAGENTSAFETY_CUSTOM_TAG}"
    if target != "binary":
        tag = f"{tag}-{target}"
    return tag


def _local_image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def build_workspace_image(
    image: str | None = None,
    target: str | None = None,
    base_image: str | None = None,
    force_rebuild: bool = False,
    no_cache: bool = False,
    push: bool = False,
) -> str:
    """Build OpenAgentSafety agent-server image.

    Args:
        image: Target image repo/name (default: EVAL_AGENT_SERVER_IMAGE).
        target: Build target for tag naming (default: source-minimal).
        base_image: Override base image used in the Dockerfile.
        force_rebuild: if True, ignore existing images and rebuild.
        no_cache: if True, pass --no-cache to docker build to avoid layer cache.
        push: if True, push the image to registry via buildx.
    """
    image_tag = resolve_openagentsafety_image_tag(
        image=image, target=target, sdk_short_sha=SDK_SHORT_SHA
    )
    base_image = base_image or OPENAGENTSAFETY_BASE_IMAGE

    if not force_rebuild:
        if push and image_exists(image_tag):
            logger.info("Using existing registry image: %s", image_tag)
            return image_tag
        if not push and _local_image_exists(image_tag):
            logger.info("Using existing local image: %s", image_tag)
            return image_tag

    logger.info("Building OpenAgentSafety image: %s", image_tag)
    logger.info("Base image: %s", base_image)
    logger.info("Push: %s", push)

    dockerfile_dir = Path(__file__).parent  # benchmarks/benchmarks/openagentsafety/
    build_context = dockerfile_dir.parent.parent.parent

    logger.info("Build context: %s", build_context)
    logger.info("Dockerfile: %s", dockerfile_dir / "Dockerfile")

    # Use shared build helper for consistent error handling and logging
    result = run_docker_build_layer(
        dockerfile=dockerfile_dir / "Dockerfile",
        context=build_context,
        tags=[image_tag],
        build_args={"BASE_IMAGE": base_image},
        push=push,
        platform="linux/amd64",
        load=not push,
        no_cache=no_cache,
    )

    if result.error:
        logger.error("Build failed: %s", result.error)
        raise RuntimeError(f"Failed to build Docker image: {result.error}")

    # Verify image exists in local docker after --load
    if not push and not _local_image_exists(image_tag):
        raise RuntimeError(
            f"Image {image_tag} was not created successfully (not present in local docker)"
        )

    logger.info("Successfully built %s", image_tag)
    return image_tag


def main(argv: list[str]) -> int:
    parser = get_build_parser()
    parser.description = "Build the OpenAgentSafety agent-server image."
    parser.add_argument(
        "--base-image",
        default=OPENAGENTSAFETY_BASE_IMAGE,
        help="Base image to use for OpenAgentSafety.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild even if the image already exists.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Docker build cache.",
    )
    args = parser.parse_args(argv)

    image_tag = resolve_openagentsafety_image_tag(
        image=args.image, target=args.target, sdk_short_sha=SDK_SHORT_SHA
    )
    logger.info("OpenAgentSafety image tag: %s", image_tag)

    if args.dry_run:
        print(image_tag)
        return 0

    build_workspace_image(
        image=args.image,
        target=args.target,
        base_image=args.base_image,
        force_rebuild=args.force_rebuild,
        no_cache=args.no_cache,
        push=args.push,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
