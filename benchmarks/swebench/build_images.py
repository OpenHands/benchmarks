#!/usr/bin/env python3
"""
Build agent-server images for all unique SWE-Bench base images in a dataset split,
optionally wrapping them with a lightweight layer that pins docutils<0.21 and installs roman.

Example:
  uv run benchmarks/swebench/build_images.py \
    --dataset princeton-nlp/SWE-bench_Verified --split test \
    --image ghcr.io/openhands/eval-agent-server --target source-minimal
"""

import sys
from pathlib import Path

from benchmarks.utils.build_utils import (
    BuildOutput,
    build_all_images,
    default_build_output_dir,
    get_build_parser,
    run_docker_build_layer,
)
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.image_utils import image_exists
from openhands.sdk import get_logger


logger = get_logger(__name__)
WRAPPER_DOCKERFILE = Path(__file__).with_name("Dockerfile.swebench-deps")
# Repos that require the docutils/roman wrapper layer
WRAPPED_REPOS = {"sphinx-doc"}


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


def should_wrap_custom_tag(custom_tag: str) -> bool:
    prefix = "sweb.eval.x86_64."
    if custom_tag.startswith(prefix):
        custom_tag = custom_tag[len(prefix) :]
    return custom_tag.split("_", 1)[0] in WRAPPED_REPOS


def should_wrap_instance_id(instance_id: str) -> bool:
    repo = instance_id.split("__")[0]
    return repo in WRAPPED_REPOS


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


def wrap_image(agent_image: str, push: bool = False) -> BuildOutput:
    """
    Wrap an agent-server image with pinned docutils/roman.

    For pushes, verify the base tag exists in the registry. For local builds,
    assume the tag is available locally or resolvable by Docker during buildx.
    """
    if push and not image_exists(agent_image):
        return BuildOutput(
            base_image=agent_image,
            tags=[],
            error=(
                f"Agent-server image {agent_image} not found in registry. "
                "Build and push it before wrapping."
            ),
        )

    if not WRAPPER_DOCKERFILE.exists():
        return BuildOutput(
            base_image=agent_image,
            tags=[],
            error=f"Wrapper Dockerfile not found at {WRAPPER_DOCKERFILE}",
        )

    logger.info("Wrapping %s in-place", agent_image)

    return run_docker_build_layer(
        dockerfile=WRAPPER_DOCKERFILE,
        context=WRAPPER_DOCKERFILE.parent,
        tags=[agent_image],
        build_args={"SDK_IMAGE": agent_image},
        push=push,
        platform="linux/amd64",
        load=not push,
    )


def _wrap_if_needed(result: BuildOutput, push: bool) -> BuildOutput:
    """
    Post-build callback that wraps images for repos that need docutils/roman.

    This is passed to build_all_images as post_build_fn, integrating wrapping
    into the main build pass with automatic retry support.
    """
    if not result.tags:
        return result

    agent_image = result.tags[0]
    # Extract custom tag from the built image tag to check if wrapping is needed
    # Format: ghcr.io/openhands/eval-agent-server:SHA-sweb.eval.x86_64.REPO_...-target
    tag_part = agent_image.split(":")[-1] if ":" in agent_image else ""
    # Remove SDK SHA prefix and target suffix to get the custom tag
    parts = tag_part.split("-", 1)
    custom_tag = parts[1].rsplit("-", 1)[0] if len(parts) > 1 else tag_part

    if not should_wrap_custom_tag(custom_tag):
        return result

    logger.info("Image %s needs wrapping, applying docutils/roman layer", agent_image)
    wrap_result = wrap_image(agent_image, push)
    if wrap_result.error:
        return BuildOutput(
            base_image=result.base_image,
            tags=result.tags,
            error=f"Wrapping failed: {wrap_result.error}",
        )

    return result


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
        post_build_fn=_wrap_if_needed,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
