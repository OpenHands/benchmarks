"""Build OpenAgentSafety Docker image from vendor/software-agent-sdk"""

import logging
import os
import subprocess
from pathlib import Path


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

    # Build command using BuildKit and --load so teh resulting image is present in local docker images
    build_cmd = [
        "docker",
        "build",
        "-f",
        str(dockerfile_dir / "Dockerfile"),
        "-t",
        image_name,
        "--platform",
        "linux/amd64",
    ]

    if no_cache:
        build_cmd.append("--no-cache")

    build_cmd.append("--load")
    build_cmd.append(str(build_context))

    logger.info(f"#### Running: {' '.join(build_cmd)}")

    # Ensure BuildKit is enabled
    env = os.environ.copy()
    env["DOCKER_BUILDKIT"] = env.get("DOCKER_BUILDKIT", "1")

    # Run build and capture output for debugging
    build_result = subprocess.run(
        build_cmd,
        capture_output=True,
        text=True,
        env=env,
    )

    # Log build output
    if build_result.stdout:
        logger.info(f"Build STDOUT:\n{build_result.stdout}")
    if build_result.stderr:
        logger.info(f"Build STDERR:\n{build_result.stderr}")

    if build_result.returncode != 0:
        logger.error(f"Build failed (return code {build_result.returncode})")
        logger.error(f"STDOUT:\n{build_result.stdout}")
        logger.error(f"STDERR:\n{build_result.stderr}")
        raise RuntimeError(f"Failed to build Docker image: {build_result.stderr}")

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
