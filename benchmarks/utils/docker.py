"""Docker utilities for benchmark evaluations."""

import logging
import os
import subprocess
import time


logger = logging.getLogger(__name__)


def ensure_docker_running() -> None:
    """Ensure Docker daemon is reachable.

    Polls for Docker connectivity (useful when DOCKER_HOST points to a sidecar
    that's still starting up). If Docker isn't available after 30 seconds,
    fails with a clear error message.

    Raises:
        RuntimeError: If Docker daemon is not reachable within 30 seconds
    """

    def _docker_info_ok() -> bool:
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    docker_host = os.environ.get("DOCKER_HOST", "local Docker daemon")
    logger.info(f"Checking Docker connectivity via {docker_host}")

    for i in range(30):  # Wait up to 30 seconds
        if _docker_info_ok():
            logger.info("Connected to Docker daemon")
            return
        if i == 0:  # First attempt
            logger.info("Waiting for Docker daemon to be available...")
        time.sleep(1)

    raise RuntimeError(
        "Could not reach Docker daemon. Please ensure Docker is running:\n"
        "  - For local development: Start Docker Desktop or run 'sudo dockerd'\n"
        f"  - For remote evaluation: Verify DOCKER_HOST={os.environ.get('DOCKER_HOST', 'not set')}"
    )
