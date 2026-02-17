"""
Resource-limited Docker workspace for SWE-fficiency benchmark.

Extends DockerWorkspace to add CPU and memory limits for parallel evaluation.
"""

import os
import subprocess
import threading
import time
import uuid
from typing import Any

from pydantic import Field, PrivateAttr

from openhands.sdk.logger import get_logger
from openhands.sdk.utils.command import execute_command
from openhands.workspace import DockerWorkspace


logger = get_logger(__name__)


def find_available_tcp_port(
    min_port: int = 30000, max_port: int = 39999, max_attempts: int = 50
) -> int:
    """Find an available TCP port in a specified range."""
    import random
    import socket

    rng = random.SystemRandom()
    ports = list(range(min_port, max_port + 1))
    rng.shuffle(ports)

    for port in ports[:max_attempts]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", port))
            return port
        except OSError:
            pass
        finally:
            sock.close()
    return -1


def check_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        time.sleep(0.1)
        return False
    finally:
        sock.close()


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

    _container_id: str | None = PrivateAttr(default=None)
    _image_name: str | None = PrivateAttr(default=None)
    _logs_thread: threading.Thread | None = PrivateAttr(default=None)
    _stop_logs: threading.Event = PrivateAttr(default_factory=threading.Event)
    # CPU group management - set by caller for automatic cleanup
    _cpu_group: list[int] | None = PrivateAttr(default=None)
    _cpu_groups_queue: Any = PrivateAttr(default=None)

    def _start_container(self, image: str, context: Any) -> None:
        """Start the Docker container with resource limits.

        Overrides parent method to add CPU and memory constraints.
        """
        # Store the image name for cleanup
        self._image_name = image

        # Determine port
        if self.host_port is None:
            self.host_port = find_available_tcp_port()
        else:
            self.host_port = int(self.host_port)

        if not check_port_available(self.host_port):
            raise RuntimeError(f"Port {self.host_port} is not available")

        if self.extra_ports:
            if not check_port_available(self.host_port + 1):
                raise RuntimeError(
                    f"Port {self.host_port + 1} is not available for VSCode"
                )
            if not check_port_available(self.host_port + 2):
                raise RuntimeError(
                    f"Port {self.host_port + 2} is not available for VNC"
                )

        # Ensure docker is available
        docker_ver = execute_command(["docker", "version"]).returncode
        if docker_ver != 0:
            raise RuntimeError(
                "Docker is not available. Please install and start "
                "Docker Desktop/daemon."
            )

        # Prepare Docker run flags
        flags: list[str] = []
        for key in self.forward_env:
            if key in os.environ:
                flags += ["-e", f"{key}={os.environ[key]}"]

        for volume in self.volumes:
            flags += ["-v", volume]
            logger.info(f"Adding volume mount: {volume}")

        ports = ["-p", f"{self.host_port}:8000"]
        if self.extra_ports:
            ports += [
                "-p",
                f"{self.host_port + 1}:8001",  # VSCode
                "-p",
                f"{self.host_port + 2}:8002",  # Desktop VNC
            ]
        flags += ports

        # Add GPU support if enabled
        if self.enable_gpu:
            flags += ["--gpus", "all"]

        # Add resource limits
        if self.cpuset_cpus is not None:
            flags += ["--cpuset-cpus", self.cpuset_cpus]
            logger.info(f"Setting cpuset-cpus: {self.cpuset_cpus}")

        if self.nano_cpus is not None:
            flags += ["--cpus", str(self.nano_cpus / 1e9)]
            logger.info(f"Setting CPU limit: {self.nano_cpus / 1e9} CPUs")

        if self.mem_limit is not None:
            flags += ["--memory", self.mem_limit]
            logger.info(f"Setting memory limit: {self.mem_limit}")

        # Run container
        run_cmd = [
            "docker",
            "run",
            "-d",
            "--platform",
            self.platform,
            "--rm",
            "--ulimit",
            "nofile=65536:65536",  # prevent "too many open files" errors
            "--name",
            f"agent-server-{uuid.uuid4()}",
            *flags,
            image,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
        proc = execute_command(run_cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to run docker container: {proc.stderr}")

        self._container_id = proc.stdout.strip()
        logger.info(f"Started container: {self._container_id}")

        # Optionally stream logs in background
        if self.detach_logs:
            self._logs_thread = threading.Thread(
                target=self._stream_docker_logs, daemon=True
            )
            self._logs_thread.start()

        # Set host for RemoteWorkspace to use
        object.__setattr__(self, "host", f"http://localhost:{self.host_port}")
        object.__setattr__(self, "api_key", None)

        # Wait for container to be healthy
        self._wait_for_health()
        logger.info(f"Docker workspace is ready at {self.host}")

        # Now initialize the parent RemoteWorkspace
        # Call the grandparent's model_post_init to avoid re-running container setup
        from openhands.sdk.workspace import RemoteWorkspace

        RemoteWorkspace.model_post_init(self, context)

    def _stream_docker_logs(self) -> None:
        """Stream Docker logs to stdout in the background."""
        import sys

        if not self._container_id:
            return
        try:
            p = subprocess.Popen(
                ["docker", "logs", "-f", self._container_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if p.stdout is None:
                return
            for line in iter(p.stdout.readline, ""):
                if self._stop_logs.is_set():
                    break
                if line:
                    sys.stdout.write(f"[DOCKER] {line}")
                    sys.stdout.flush()
        except Exception as e:
            import sys

            sys.stderr.write(f"Error streaming docker logs: {e}\n")
        finally:
            try:
                self._stop_logs.set()
            except Exception:
                pass

    def _wait_for_health(self, timeout: float = 120.0) -> None:
        """Wait for the Docker container to become healthy."""
        from urllib.request import urlopen

        start = time.time()
        health_url = f"http://127.0.0.1:{self.host_port}/health"

        while time.time() - start < timeout:
            try:
                with urlopen(health_url, timeout=1.0) as resp:
                    if 200 <= getattr(resp, "status", 200) < 300:
                        return
            except Exception:
                pass

            # Check if container is still running
            if self._container_id:
                ps = execute_command(
                    [
                        "docker",
                        "inspect",
                        "-f",
                        "{{.State.Running}}",
                        self._container_id,
                    ]
                )
                if ps.stdout.strip() != "true":
                    logs = execute_command(["docker", "logs", self._container_id])
                    msg = (
                        "Container stopped unexpectedly. Logs:\n"
                        f"{logs.stdout}\n{logs.stderr}"
                    )
                    raise RuntimeError(msg)
            time.sleep(1)
        raise RuntimeError("Container failed to become healthy in time")

    def cleanup(self) -> None:
        """Stop and remove the Docker container, and return CPU group to queue."""
        if self._container_id:
            # Stop logs streaming
            self._stop_logs.set()
            if self._logs_thread and self._logs_thread.is_alive():
                self._logs_thread.join(timeout=2)

            # Stop and remove the container
            logger.info(f"Stopping container: {self._container_id}")
            execute_command(["docker", "stop", self._container_id])
            self._container_id = None

        # Optionally delete the Docker image
        if self.cleanup_image and self._image_name:
            logger.info(f"Deleting Docker image: {self._image_name}")
            result = execute_command(["docker", "rmi", "-f", self._image_name])
            if result.returncode == 0:
                logger.info(f"Successfully deleted image: {self._image_name}")
            else:
                logger.warning(
                    f"Failed to delete image {self._image_name}: {result.stderr}"
                )
            self._image_name = None

        # Return CPU group to queue if set
        if self._cpu_groups_queue is not None and self._cpu_group is not None:
            try:
                self._cpu_groups_queue.put(self._cpu_group)
                logger.info(f"Returned CPU group to queue: {self._cpu_group}")
            except Exception as e:
                logger.warning(f"Failed to return CPU group to queue: {e}")
            self._cpu_group = None
