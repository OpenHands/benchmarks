#!/usr/bin/env python3
"""
Build agent-server images for all unique SWE-Bench base images in a dataset split,
optionally wrapping them with a lightweight layer that pins docutils<0.21 and installs roman.

The build uses a two-phase "split" strategy to avoid full 9h+ rebuilds when only
the SDK commit changes:

  Phase 1 – eval-base images (SDK-independent, cached across SDK commits)
    Tag: ghcr.io/openhands/eval-base:v1-sweb.eval.x86_64.django_1776_django-12155
    Contains: SWE-bench base + apt/npm packages + user setup.
    Only rebuilt when the SWE-bench upstream image or the base Dockerfile changes.

  Phase 2 – agent-server images (thin layer, seconds per image)
    Tag: ghcr.io/openhands/eval-agent-server:<SDK_SHA>-<custom_tag>-source-minimal
    Contains: pre-built base + SDK venv (COPY from builder stage).

Example:
  uv run benchmarks/swebench/build_images.py \
    --dataset princeton-nlp/SWE-bench_Verified --split test \
    --image ghcr.io/openhands/eval-agent-server --target source-minimal
"""

import functools
import sys
from pathlib import Path

from benchmarks.swebench import constants
from benchmarks.swebench.config import BUILD_DEFAULTS
from benchmarks.utils.build_utils import (
    BuildOutput,
    build_agent_layer,
    build_all_base_images,
    build_all_images,
    default_build_output_dir,
    get_build_parser,
    run_docker_build_layer,
)
from benchmarks.utils.constants import EVAL_BASE_IMAGE
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.image_utils import remote_image_exists
from openhands.sdk import get_logger


logger = get_logger(__name__)
WRAPPER_DOCKERFILE = Path(__file__).with_name("Dockerfile.swebench-deps")


def get_official_docker_image(
    instance_id: str,
    docker_image_prefix: str = constants.DOCKER_IMAGE_PREFIX,
) -> str:
    # Official SWE-Bench image
    # swebench/sweb.eval.x86_64.django_1776_django-11333:v1
    repo, name = instance_id.split("__")
    official_image_name = docker_image_prefix.rstrip("/")
    official_image_name += (
        f"/sweb.eval.x86_64.{repo}_1776_{name}:{constants.DOCKER_IMAGE_TAG}".lower()
    )
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


def eval_base_tag_for_custom_tag(custom_tag: str) -> str:
    """Return the full eval-base image reference for a given custom tag."""
    return f"{EVAL_BASE_IMAGE}:{constants.BASE_IMAGE_TAG_VERSION}-{custom_tag}"


def should_wrap_custom_tag(custom_tag: str) -> bool:
    prefix = "sweb.eval.x86_64."
    if custom_tag.startswith(prefix):
        custom_tag = custom_tag[len(prefix) :]
    return custom_tag.split("_", 1)[0] in constants.WRAPPED_REPOS


def should_wrap_instance_id(instance_id: str) -> bool:
    repo = instance_id.split("__")[0]
    return repo in constants.WRAPPED_REPOS


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
    if push and not remote_image_exists(agent_image):
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


def _agent_layer_build(
    base_image: str,
    target_image: str,
    custom_tag: str,
    target="source-minimal",
    push: bool = False,
    force_build: bool = False,
    cached_sdist=None,
    *,
    _mapping: dict[str, str],
) -> BuildOutput:
    """Build an agent layer image by looking up the eval-base for *base_image*."""
    eval_base = _mapping.get(base_image)
    if eval_base is None:
        return BuildOutput(
            base_image=base_image,
            tags=[],
            error=f"No eval-base mapping for {base_image}",
            status="failed",
        )
    return build_agent_layer(
        base_image=eval_base,
        target_image=target_image,
        custom_tag=custom_tag,
        target=target,
        push=push,
        force_build=force_build,
        cached_sdist=cached_sdist,
    )


def _make_agent_layer_build_fn(
    swebench_image_to_eval_base: dict[str, str],
) -> functools.partial:
    """
    Create a build function that maps SWE-bench base images to pre-built
    eval-base images and delegates to ``build_agent_layer``.

    ``build_all_images`` passes ``base_image`` (the SWE-bench image) to the
    build function.  We intercept it, look up the corresponding eval-base tag,
    and forward to ``build_agent_layer``.

    Returns a ``functools.partial`` (picklable) so that
    ``ProcessPoolExecutor`` in ``build_all_images`` can serialize it.
    """
    return functools.partial(_agent_layer_build, _mapping=swebench_image_to_eval_base)


def main(argv: list[str]) -> int:
    parser = get_build_parser()
    parser.set_defaults(**BUILD_DEFAULTS)
    args = parser.parse_args(argv)

    base_images: list[str] = collect_unique_base_images(
        args.dataset,
        args.split,
        args.n_limit,
        args.select,
    )
    build_dir = default_build_output_dir(args.dataset, args.split)

    if args.dry_run:
        print("\n".join(base_images))
        return 0

    # ── Phase 1: build eval-base images (SDK-independent, cached across commits) ──
    logger.info(
        "Phase 1: building %d eval-base images (SDK-independent)", len(base_images)
    )
    base_rc = build_all_base_images(
        base_images=base_images,
        base_image_to_custom_tag_fn=extract_custom_tag,
        base_image_tag_version=constants.BASE_IMAGE_TAG_VERSION,
        build_dir=build_dir,
        push=args.push,
        max_workers=args.max_workers,
        force_build=args.force_build,
        max_retries=args.max_retries,
    )
    if base_rc != 0:
        logger.error("Phase 1 failed: some base images could not be built.")
        return base_rc

    # ── Phase 2: build thin agent layers on pre-built bases ──
    logger.info("Phase 2: building %d agent layers on cached bases", len(base_images))

    # Build mapping: SWE-bench image → eval-base tag
    swebench_to_eval_base = {
        img: eval_base_tag_for_custom_tag(extract_custom_tag(img))
        for img in base_images
    }

    return build_all_images(
        base_images=base_images,
        target=args.target,
        build_dir=build_dir,
        image=args.image,
        push=args.push,
        max_workers=args.max_workers,
        build_batch_size=args.build_batch_size,
        dry_run=False,
        force_build=args.force_build,
        max_retries=args.max_retries,
        base_image_to_custom_tag_fn=extract_custom_tag,
        post_build_fn=_wrap_if_needed,
        build_fn=_make_agent_layer_build_fn(swebench_to_eval_base),
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
