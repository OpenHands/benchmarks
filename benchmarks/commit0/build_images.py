#!/usr/bin/env python3
"""
Build agent-server images for Commit0 repositories.

Example:
  uv run benchmarks/commit0/build_images.py \
    --dataset wentingzhao/commit0_combined --split test --repo-split lite \
    --image ghcr.io/openhands/eval-agent-server --push --max-workers 16
"""

import os
import sys

from commit0.harness.constants import SPLIT

from benchmarks.utils.build_utils import (
    apply_cache_mode,
    build_all_images,
    default_build_output_dir,
    get_build_parser,
)
from openhands.sdk import get_logger


logger = get_logger(__name__)
DEFAULT_DOCKER_IMAGE_PREFIX = "docker.io/wentingzhao/"


def get_base_docker_image(
    repo_name: str,
    docker_image_prefix: str | None = None,
) -> str:
    """Get the upstream Commit0 base image for a repository."""
    prefix = docker_image_prefix or os.getenv(
        "EVAL_DOCKER_IMAGE_PREFIX", DEFAULT_DOCKER_IMAGE_PREFIX
    )
    return (prefix.rstrip("/") + "/" + repo_name).lower() + ":v0"


def extract_custom_tag(base_image: str) -> str:
    """Extract Commit0 custom tag from a base image name."""
    repo_tag = base_image.rsplit("/", 1)[-1]
    repo_name = repo_tag.split(":", 1)[0].lower()
    return f"commit0-{repo_name}"


def _load_selected_instances(selected_instances_file: str) -> list[str]:
    selected: list[str] = []
    with open(selected_instances_file, "r", encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if name:
                selected.append(name)
    return selected


def resolve_repos(repo_split: str) -> list[str]:
    """Resolve repository names for a Commit0 repo split."""
    repo_split = repo_split.strip()
    if repo_split in SPLIT:
        repos = list(SPLIT[repo_split])
    else:
        repos = [repo_split]
    return repos


def collect_base_images(
    repo_split: str,
    n_limit: int,
    selected_instances_file: str | None,
    docker_image_prefix: str | None,
) -> list[str]:
    repos = resolve_repos(repo_split)

    if selected_instances_file:
        selected = set(_load_selected_instances(selected_instances_file))
        repos = [repo for repo in repos if repo in selected]

    if n_limit:
        repos = repos[:n_limit]

    if not repos:
        raise ValueError("No Commit0 repositories selected for image build")

    logger.info("Preparing %d Commit0 repos for build", len(repos))
    return [get_base_docker_image(repo, docker_image_prefix) for repo in repos]


def main(argv: list[str]) -> int:
    parser = get_build_parser()
    parser.add_argument(
        "--repo-split",
        type=str,
        default="lite",
        help="Commit0 repo split (lite, all, or repo name)",
    )
    parser.add_argument(
        "--docker-image-prefix",
        type=str,
        default="",
        help="Override base image prefix (default: env EVAL_DOCKER_IMAGE_PREFIX)",
    )
    parser.set_defaults(dataset="wentingzhao/commit0_combined")
    args = parser.parse_args(argv)
    apply_cache_mode(args)

    docker_image_prefix = args.docker_image_prefix or None

    base_images = collect_base_images(
        repo_split=args.repo_split,
        n_limit=args.n_limit,
        selected_instances_file=args.select,
        docker_image_prefix=docker_image_prefix,
    )

    build_dir = default_build_output_dir(args.dataset, args.split)
    return build_all_images(
        base_images=base_images,
        target=args.target,
        build_dir=build_dir,
        image=args.image,
        push=args.push,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        base_image_to_custom_tag_fn=extract_custom_tag,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
