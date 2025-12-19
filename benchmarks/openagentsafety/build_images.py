"""Build OpenAgentSafety Docker image from vendor/software-agent-sdk"""

import logging
import subprocess
from pathlib import Path

from benchmarks.utils.build_utils import run_docker_build_layer


logger = logging.getLogger(__name__)


def get_vendor_sdk_commit() -> str:
    """Get the commit hash of the vendor SDK."""
    repo_root = Path(__file__).parent.parent.parent
    vendor_sdk_path = repo_root / "vendor" / "software-agent-sdk"

    if not vendor_sdk_path.exists():
        raise RuntimeError(f"Vendor SDK not found at {vendor_sdk_path}")

    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=vendor_sdk_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to get SDK commit: {result.stderr}")

    return result.stdout.strip()


def check_image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def build_workspace_image(force_rebuild: bool = False, no_cache: bool = False) -> str:
    """Build Docker image using SDK from vendor folder.

    Args:
        force_rebuild: if True, ignore existing images and rebuild.
        no_cache: if True, pass --no-cache to docker build to avoid layer cache.
    """
    sdk_commit = get_vendor_sdk_commit()
    image_name = f"openagentsafety-agent-server:{sdk_commit}"

    if not force_rebuild and check_image_exists(image_name):
        logger.info(f"#### Using existing image: {image_name}")
        return image_name

    logger.info(f"#### Building Docker image: {image_name}")
    logger.info(f"#### SDK version: {sdk_commit}")
    logger.info("#### This will take approximately 3-5 minutes...")

    dockerfile_dir = Path(__file__).parent  # benchmarks/benchmarks/openagentsafety/
    build_context = dockerfile_dir.parent.parent.parent

    logger.info(f"Build context: {build_context}")
    logger.info(f"Dockerfile: {dockerfile_dir / 'Dockerfile'}")

    # Use shared build helper for consistent error handling and logging
    result = run_docker_build_layer(
        dockerfile=dockerfile_dir / "Dockerfile",
        context=build_context,
        tags=[image_name],
        build_args=None,
        push=False,
        platform="linux/amd64",
        load=True,
        no_cache=no_cache,
    )

    if result.error:
        logger.error(f"Build failed: {result.error}")
        raise RuntimeError(f"Failed to build Docker image: {result.error}")

    # Verify image exists in local docker after --load
    if not check_image_exists(image_name):
        raise RuntimeError(
            f"Image {image_name} was not created successfully (not present in local docker)"
        )

    logger.info(f"#### Successfully built {image_name}")
    return image_name


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    image = build_workspace_image(force_rebuild=True, no_cache=False)
    print(f"Image ready: {image}")
