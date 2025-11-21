#!/usr/bin/env python3
import base64
import sys

import requests


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


def image_exists(
    image_ref: str,
    gh_username: str | None = None,
    gh_pat: str | None = None,  # GitHub PAT with read:packages for private GHCR
    docker_token: str | None = None,  # Docker Hub JWT if you already have one
) -> bool:
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

    ok = image_exists(image, gh_username=gh_user, gh_pat=gh_pat)
    print(f"{image} -> {'✅ exists' if ok else '❌ not found or unauthorized'}")
