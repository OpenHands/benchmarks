#!/usr/bin/env python3
"""
Build agent-server images for all unique SWE-Bench base images in a dataset split.

Example:
  uv run benchmarks/swebench/build_images.py \
    --dataset princeton-nlp/SWE-bench_Verified --split test \
    --image ghcr.io/openhands/eval-agent-server --target source-minimal
"""

import subprocess
import sys
from pathlib import Path

from benchmarks.utils.build_utils import (
    _get_sdk_submodule_info,
    build_all_images,
    default_build_output_dir,
    get_build_parser,
)
from benchmarks.utils.dataset import get_dataset
from openhands.sdk import get_logger


logger = get_logger(__name__)


def get_official_docker_image(
    instance_id: str,
    docker_image_prefix="docker.io/swebench/",
) -> str:
    # Official SWE-Bench image
    # swebench/sweb.eval.x86_64.django_1776_django-11333:v1
    repo, name = instance_id.split("__")
    official_image_name = docker_image_prefix.rstrip("/")
    official_image_name += f"/sweb.eval.x86_64.{repo}_1776_{name}:latest".lower()
    logger.debug(f"Official SWE-Bench image: {official_image_name}")
    return official_image_name


def extract_custom_tag(base_image: str) -> str:
    """
    Extract SWE-Bench instance ID from official SWE-Bench image name.

    Example:
        docker.io/swebench/sweb.eval.x86_64.django_1776_django-12155:latest
        -> sweb.eval.x86_64.django_1776_django-12155
    """
    name_tag = base_image.split("/")[-1]
    name = name_tag.split(":")[0]
    return name


def collect_unique_base_images(
    dataset,
    split,
    n_limit,
    selected_instances_file: str | None = None,
):
    df = get_dataset(
        dataset_name=dataset,
        split=split,
        eval_limit=n_limit if n_limit else None,
        selected_instances_file=selected_instances_file,
    )
    return sorted(
        {get_official_docker_image(str(row["instance_id"])) for _, row in df.iterrows()}
    )


def wrap_images_with_deps_fix(
    base_images: list[str],
    sdk_sha: str,
    target: str,
    push: bool = False,
) -> list[str]:
    """
    Wrap base SDK images with docutils/roman dependency fix.
    Returns list of wrapped image tags.
    """
    dockerfile_path = Path(__file__).parent / "Dockerfile.swebench-deps"
    if not dockerfile_path.exists():
        logger.warning(
            "Dockerfile.swebench-deps not found, skipping dependency fix wrapping"
        )
        return []

    wrapped_tags = []
    for base_image in base_images:
        # Extract custom tag from base image (e.g., "sweb.eval.x86_64.django_1776_django-12155")
        custom_tag = extract_custom_tag(base_image)

        # Construct base and wrapped image names
        # Base: ghcr.io/openhands/eval-agent-server:abc1234-sweb.eval.x86_64.django_1776_django-12155-source-minimal
        # Wrapped: ghcr.io/openhands/eval-agent-server:abc1234-sweb.eval.x86_64.django_1776_django-12155-source-minimal-fixed
        base_tag = (
            f"ghcr.io/openhands/eval-agent-server:{sdk_sha}-{custom_tag}-{target}"
        )
        wrapped_tag = f"{base_tag}-fixed"

        logger.info(f"Wrapping {base_tag} -> {wrapped_tag}")

        # Build wrapped image
        cmd = [
            "docker",
            "build",
            "-f",
            str(dockerfile_path),
            "--build-arg",
            f"SDK_IMAGE={base_tag}",
            "-t",
            wrapped_tag,
            ".",
        ]

        try:
            subprocess.check_call(cmd, cwd=Path(__file__).parent.parent.parent)
            wrapped_tags.append(wrapped_tag)

            if push:
                logger.info(f"Pushing {wrapped_tag}")
                subprocess.check_call(["docker", "push", wrapped_tag])

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to wrap {base_tag}: {e}")
            continue

    return wrapped_tags


def main(argv: list[str]) -> int:
    parser = get_build_parser()
    args = parser.parse_args(argv)

    base_images: list[str] = collect_unique_base_images(
        args.dataset,
        args.split,
        args.n_limit,
        args.select,
    )
    build_dir = default_build_output_dir(args.dataset, args.split)

    # Build base SDK images
    result = build_all_images(
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

    if result != 0 or args.dry_run:
        return result

    # Wrap images with dependency fix
    git_ref, git_sha, sdk_version = _get_sdk_submodule_info()
    wrapped_tags = wrap_images_with_deps_fix(
        base_images=base_images,
        sdk_sha=git_sha[:7],
        target=args.target,
        push=args.push,
    )

    if wrapped_tags:
        logger.info(
            f"Successfully wrapped {len(wrapped_tags)} images with dependency fix"
        )
    else:
        logger.warning("No images were wrapped with dependency fix")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
