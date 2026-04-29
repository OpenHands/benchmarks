#!/usr/bin/env python3
from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from openhands.sdk.workspace import TargetType
    from openhands.workspace import (
        ApptainerWorkspace,
        DockerDevWorkspace,
        DockerWorkspace,
    )

import requests

from openhands.sdk import get_logger


logger = get_logger(__name__)


ACCEPT = ",".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    ]
)


def _parse(image: str):
    digest = None
    if "@" in image:
        image, digest = image.split("@", 1)
    tag = None
    last = image.rsplit("/", 1)[-1]
    if ":" in last:  # tag after last slash (not registry:port)
        image, tag = image.rsplit(":", 1)
    parts = image.split("/")
    if "." in parts[0] or ":" in parts[0] or parts[0] == "localhost":
        registry, repo = parts[0], "/".join(parts[1:])
    else:
        registry, repo = "registry-1.docker.io", "/".join(parts)
    ref = digest or tag or "latest"
    return registry, repo, ref


def _dockerhub_token(repo: str) -> str | None:
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
    r = requests.get(url, timeout=10)
    if r.ok:
        return r.json().get("token")
    return None


def _ghcr_token(repo: str, username: str | None, pat: str | None) -> str | None:
    # Public: anonymous works; Private: Basic auth with PAT (read:packages) to get bearer
    url = f"https://ghcr.io/token?service=ghcr.io&scope=repository:{repo}:pull"
    headers = {}
    if username and pat:
        headers["Authorization"] = (
            "Basic " + base64.b64encode(f"{username}:{pat}".encode()).decode()
        )
    r = requests.get(url, headers=headers, timeout=10)
    if r.ok:
        return r.json().get("token")
    return None


def local_image_exists(image: str) -> bool:
    """Check if a Docker image exists in the local Docker daemon."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            check=False,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Failed to check if image {image} exists: {e}")
        return False


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def get_apptainer_cache_dir() -> str:
    """Return the Apptainer cache directory used by the workspace."""
    return os.getenv("APPTAINER_CACHE_DIR") or str(Path.home() / ".apptainer_cache")


def get_apptainer_sif_path(
    agent_server_image: str, cache_dir: str | None = None
) -> str:
    """Return the cached SIF path ApptainerWorkspace would use for an image."""
    resolved_cache_dir = cache_dir or get_apptainer_cache_dir()
    sif_name = agent_server_image.replace(":", "_").replace("/", "_") + ".sif"
    return str(Path(resolved_cache_dir) / sif_name)


def create_apptainer_workspace(
    agent_server_image: str,
    working_dir: str = "/workspace",
    forward_env: list[str] | None = None,
    extra_ports: bool = False,
) -> ApptainerWorkspace:
    """Create an Apptainer workspace from a pre-built agent-server image.

    Unlike DockerDevWorkspace, ApptainerWorkspace cannot build images from a
    base image on the fly. The image must already exist in a container registry
    that `apptainer pull docker://...` can access, or the corresponding SIF file
    must already be present in the configured Apptainer cache.
    """
    from openhands.workspace import ApptainerWorkspace

    host_port = os.getenv("APPTAINER_HOST_PORT")
    cache_dir = get_apptainer_cache_dir()
    mount_dir = os.getenv("APPTAINER_MOUNT_DIR")
    sif_path = get_apptainer_sif_path(agent_server_image, cache_dir)

    workspace_kwargs = {
        "working_dir": working_dir,
        "forward_env": forward_env or [],
        "extra_ports": extra_ports,
        "host_port": int(host_port) if host_port else None,
        "cache_dir": cache_dir,
        "mount_dir": mount_dir or None,
        "use_fakeroot": _env_flag("APPTAINER_USE_FAKEROOT", True),
        "enable_docker_compat": _env_flag("APPTAINER_ENABLE_DOCKER_COMPAT", True),
    }

    if Path(sif_path).exists():
        logger.info(
            "Using cached Apptainer SIF %s for image %s", sif_path, agent_server_image
        )
        return ApptainerWorkspace(sif_file=sif_path, **workspace_kwargs)

    if not remote_image_exists(agent_server_image):
        raise RuntimeError(
            f"Agent server image {agent_server_image} does not exist in container registry. "
            "Apptainer can only use a registry-pullable image or an existing cached SIF file. "
            "If you built images with a benchmark build_images.py script, re-run it with --push "
            "from a Docker-capable machine or CI; local-only builds are not enough. "
            "If the images were built from a different checkout, make sure IMAGE_TAG_PREFIX "
            "matches the tag prefix used during the build."
        )

    logger.info(f"Using Apptainer workspace with image {agent_server_image}")
    return ApptainerWorkspace(server_image=agent_server_image, **workspace_kwargs)


def create_docker_workspace(
    agent_server_image: str,
    base_image: str,
    build_target: TargetType,
    working_dir: str = "/workspace",
    forward_env: list[str] | None = None,
) -> DockerWorkspace | DockerDevWorkspace:
    """Create a Docker workspace, building the image only if not already available.

    Returns DockerWorkspace when a pre-built image is found locally,
    DockerDevWorkspace otherwise (which builds on-the-fly).
    Set FORCE_BUILD=1 to skip auto-detection and always build.
    """
    from openhands.workspace import DockerDevWorkspace, DockerWorkspace

    force_build = os.getenv("FORCE_BUILD", "0").lower() in ("1", "true", "yes")
    if not force_build and local_image_exists(agent_server_image):
        logger.info(f"Using pre-built image {agent_server_image}")
        return DockerWorkspace(
            server_image=agent_server_image,
            working_dir=working_dir,
            forward_env=forward_env or [],
        )
    else:
        if force_build:
            logger.info(f"FORCE_BUILD set, building workspace from {base_image}...")
        else:
            logger.info(f"Building workspace from {base_image}...")
        return DockerDevWorkspace(
            base_image=base_image,
            working_dir=working_dir,
            target=build_target,
            forward_env=forward_env or [],
        )


def remote_image_exists(
    image_ref: str,
    gh_username: str | None = None,
    gh_pat: str | None = None,  # GitHub PAT with read:packages for private GHCR
    docker_token: str | None = None,  # Docker Hub JWT if you already have one
) -> bool:
    """Check if a Docker image exists in a remote registry."""
    registry, repo, ref = _parse(image_ref)
    headers = {"Accept": ACCEPT}

    if registry in ("docker.io", "index.docker.io", "registry-1.docker.io"):
        base = "https://registry-1.docker.io"
        token = docker_token or _dockerhub_token(repo)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    elif registry == "ghcr.io":
        base = "https://ghcr.io"
        token = _ghcr_token(repo, gh_username, gh_pat)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    else:
        base = f"https://{registry}"

    url = f"{base}/v2/{repo}/manifests/{ref}"
    try:
        r = requests.head(url, headers=headers, timeout=10)
        if r.status_code in (
            405,
            406,
        ):  # some registries disallow HEAD or need GET for content-negotiation
            r = requests.get(url, headers=headers, timeout=10)
        # 200 -> exists; 401/403 -> exists but unauthorized; 404 -> not found
        return r.status_code == 200
    except requests.RequestException:
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python image_check.py <image[:tag]|image@sha256:...> [gh_user] [gh_pat]"
        )
        sys.exit(1)

    image = sys.argv[1]
    gh_user = sys.argv[2] if len(sys.argv) > 2 else None
    gh_pat = sys.argv[3] if len(sys.argv) > 3 else None

    ok = remote_image_exists(image, gh_username=gh_user, gh_pat=gh_pat)
    print(f"{image} -> {'✅ exists' if ok else '❌ not found or unauthorized'}")
