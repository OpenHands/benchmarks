"""
Resource-limited Docker workspace for SWE-fficiency benchmark.

Extends DockerWorkspace to add CPU and memory limits for parallel evaluation.
"""

from typing import Any

from pydantic import Field, PrivateAttr

from openhands.sdk.logger import get_logger
from openhands.sdk.utils.command import execute_command
from openhands.workspace import DockerWorkspace


logger = get_logger(__name__)


class ResourceLimitedDockerWorkspace(DockerWorkspace):
    """DockerWorkspace with CPU and memory resource limits.

    This workspace extends DockerWorkspace to support CPU pinning and memory
    limits for parallel evaluation scenarios where resource isolation is needed.

    Example:
        with ResourceLimitedDockerWorkspace(
            server_image="ghcr.io/swefficiency/swefficiency-images:instance-id",
            cpuset_cpus="0,1,2,3",
            mem_limit="16g",
        ) as workspace:
            result = workspace.execute_command("python benchmark.py")
    """

    # Resource limit configuration
    cpuset_cpus: str | None = Field(
        default=None,
        description="CPUs to use (e.g., '0,1,2,3' or '0-3'). If None, no CPU pinning.",
    )
    nano_cpus: int | None = Field(
        default=None,
        description="CPU quota in nanoseconds (1e9 = 1 CPU). If None, unlimited.",
    )
    mem_limit: str | None = Field(
        default="16g",
        description="Memory limit (e.g., '16g', '8192m'). If None, unlimited.",
    )

    # CPU group management - set by caller for automatic cleanup
    _cpu_group: list[int] | None = PrivateAttr(default=None)
    _cpu_groups_queue: Any = PrivateAttr(default=None)

    def _start_container(self, image: str, context: Any) -> None:
        """Start the Docker container with resource limits.

        Delegates to parent for container lifecycle, then applies
        CPU and memory constraints via docker update.
        """
        super()._start_container(image, context)
        self._apply_resource_limits()

    def _apply_resource_limits(self) -> None:
        """Apply CPU and memory limits to the running container."""
        if not self._container_id:
            return

        update_flags: list[str] = []

        if self.cpuset_cpus is not None:
            update_flags += ["--cpuset-cpus", self.cpuset_cpus]
            logger.info(f"Setting cpuset-cpus: {self.cpuset_cpus}")

        if self.nano_cpus is not None:
            update_flags += ["--cpus", str(self.nano_cpus / 1e9)]
            logger.info(f"Setting CPU limit: {self.nano_cpus / 1e9} CPUs")

        if self.mem_limit is not None:
            update_flags += ["--memory", self.mem_limit]
            logger.info(f"Setting memory limit: {self.mem_limit}")

        if update_flags:
            result = execute_command(
                ["docker", "update", *update_flags, self._container_id]
            )
            if result.returncode != 0:
                logger.warning(f"Failed to apply resource limits: {result.stderr}")

    def cleanup(self) -> None:
        """Stop and remove the Docker container, and return CPU group to queue."""
        super().cleanup()

        # Return CPU group to queue if set
        if self._cpu_groups_queue is not None and self._cpu_group is not None:
            try:
                self._cpu_groups_queue.put(self._cpu_group)
                logger.info(f"Returned CPU group to queue: {self._cpu_group}")
            except Exception as e:
                logger.warning(f"Failed to return CPU group to queue: {e}")
            self._cpu_group = None
