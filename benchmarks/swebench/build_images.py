#!/usr/bin/env python3
"""
Build agent-server images for all unique SWE-Bench base images in a dataset split.

Uses a three-phase build pipeline:
  1. Build shared builder image (SDK + dependencies)
  2. Build per-instance base images
  3. Assemble final agent images locally

Example:
  uv run benchmarks/swebench/build_images.py \
    --dataset princeton-nlp/SWE-bench_Verified --split test \
    --image ghcr.io/openhands/eval-agent-server
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.swebench import constants
from benchmarks.utils.build_utils import (
    BuildOutput,
    default_build_output_dir,
    run_docker_build_layer,
)
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.image_utils import local_image_exists, remote_image_exists
from benchmarks.utils.version import get_phased_image_tag_prefix
from openhands.sdk import get_logger


logger = get_logger(__name__)
WRAPPER_DOCKERFILE = Path(__file__).with_name("Dockerfile.swebench-deps")
GRADING_ENTRYPOINT_DOCKERFILE = Path(__file__).with_name(
    "Dockerfile.grading-entrypoint"
)


def get_official_docker_image(
    instance_id: str,
    docker_image_prefix: str = constants.DOCKER_IMAGE_PREFIX,
) -> str:
    image_template = os.getenv("OPENHANDS_SWEBENCH_IMAGE_TEMPLATE")
    if image_template:
        repo, name = instance_id.split("__")
        return image_template.format(
            instance_id=instance_id,
            repo=repo,
            name=name,
            arch="x86_64",
        )

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


def prepare_local_grading_image(agent_image: str, grading_image: str) -> BuildOutput:
    """
    Alias a locally-built agent-server image under the official SWE-Bench
    grading image tag, with its ENTRYPOINT cleared.

    This lets `swebench-eval --no-modal` reuse an already-built local
    eval-agent-server image as the SWE-Bench grading image for that
    instance, instead of pulling/building the (functionally equivalent)
    official image from Docker Hub. See Dockerfile.grading-entrypoint for
    why the ENTRYPOINT needs clearing.

    Requires `agent_image` to already exist locally; this function never
    builds or pulls the agent-server image itself.

    Deliberately uses plain `docker build` rather than
    run_docker_build_layer()'s `docker buildx build`: this is a local-only
    alias (never pushed), and buildx builds route through whichever buildx
    builder instance is currently active. A `docker-container` driver
    instance (e.g. this repo's "openhands-builder", used for the phased
    image pipeline) runs BuildKit in its own container with its own image
    store, isolated from the host Docker daemon, so it can't resolve a
    locally-built-only base image as FROM. Plain `docker build` always uses
    the host daemon's own image store and has no such isolation.
    """
    if not local_image_exists(agent_image):
        return BuildOutput(
            base_image=agent_image,
            tags=[],
            error=f"Agent-server image {agent_image} not found locally.",
            status="failed",
        )

    if not GRADING_ENTRYPOINT_DOCKERFILE.exists():
        return BuildOutput(
            base_image=agent_image,
            tags=[],
            error=(
                "Grading entrypoint Dockerfile not found at "
                f"{GRADING_ENTRYPOINT_DOCKERFILE}"
            ),
        )

    logger.info("Preparing local grading image %s from %s", grading_image, agent_image)

    cmd = [
        "docker",
        "build",
        "--file",
        str(GRADING_ENTRYPOINT_DOCKERFILE),
        "--build-arg",
        f"SDK_IMAGE={agent_image}",
        "--tag",
        grading_image,
        str(GRADING_ENTRYPOINT_DOCKERFILE.parent),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return BuildOutput(
            base_image=agent_image,
            tags=[],
            error=(
                f"docker build failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            ),
            status="failed",
        )

    return BuildOutput(base_image=agent_image, tags=[grading_image], error=None)


def prepare_local_grading_images_for_predictions(
    predictions_file: Path,
) -> dict[str, bool]:
    """
    Prepare local SWE-Bench grading images for every instance in a
    predictions file, reusing already-built local eval-agent-server images
    where available.

    For each instance:
      - If the official SWE-Bench grading image already exists locally,
        nothing to do (skipped, counted as not-prepared).
      - Else, if a matching local eval-agent-server image exists (same tag
        prefix / build target run_infer.py would have built), alias it
        under the official grading image tag via
        prepare_local_grading_image().
      - Else, nothing to prepare; the instance is left for SWE-Bench's
        normal pull/build behavior during grading.

    This is purely a local speed/bandwidth optimization for `swebench-eval
    --no-modal`; it has no effect on correctness, since the grading
    environment itself (everything but ENTRYPOINT) is identical to the
    official image the eval-agent-server image was built from.

    Returns a dict mapping instance_id -> whether an image was prepared.
    """
    prefix = get_phased_image_tag_prefix()
    target = constants.DEFAULT_BUILD_TARGET
    suffix = f"-{target}" if target != constants.BUILD_TARGET_BINARY else ""

    instance_ids: list[str] = []
    with open(predictions_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            instance_id = data.get("instance_id")
            if instance_id:
                instance_ids.append(instance_id)

    results: dict[str, bool] = {}
    for instance_id in instance_ids:
        grading_image = get_official_docker_image(instance_id)
        if local_image_exists(grading_image):
            results[instance_id] = False
            continue

        custom_tag = extract_custom_tag(grading_image)
        agent_image = f"{EVAL_AGENT_SERVER_IMAGE}:{prefix}-{custom_tag}{suffix}"
        if not local_image_exists(agent_image):
            results[instance_id] = False
            continue

        output = prepare_local_grading_image(agent_image, grading_image)
        results[instance_id] = output.error is None
        if output.error:
            logger.warning(
                "Failed to prepare local grading image for %s: %s",
                instance_id,
                output.error,
            )

    return results


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build agent-server images using the three-phase pipeline."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Verified",
        help="Dataset name",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--image",
        default=EVAL_AGENT_SERVER_IMAGE,
        help="Target repo/name for final agent images",
    )
    parser.add_argument(
        "--target",
        default="source-minimal",
        help="Final image target tag suffix",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push built images to the registry",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=12,
        help="Concurrent builds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per image build",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=0,
        help="Limit number of images (0 = no limit)",
    )
    parser.add_argument(
        "--select",
        type=str,
        default=None,
        help="Path to text file containing instance IDs to select",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Rebuild final images even if matching remote tags already exist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from benchmarks.swebench.build_base_images import (
        assemble_all_agent_images,
        build_all_base_images,
        build_builder_image,
    )

    parser = get_parser()
    args = parser.parse_args(argv)

    base_images = collect_unique_base_images(
        args.dataset,
        args.split,
        args.n_limit,
        args.select,
    )
    build_dir = default_build_output_dir(args.dataset, args.split)

    builder_result = build_builder_image(push=args.push, force_build=args.force_build)
    if builder_result.error or not builder_result.tags:
        print(
            builder_result.error or "Builder image build produced no tags",
            file=sys.stderr,
        )
        return 1

    rc = build_all_base_images(
        base_images=base_images,
        build_dir=build_dir,
        push=args.push,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        force_build=args.force_build,
    )
    if rc != 0:
        return rc

    def custom_tag_fn(base: str) -> str:
        return extract_custom_tag(base)

    return assemble_all_agent_images(
        base_images=base_images,
        builder_tag=builder_result.tags[0],
        build_dir=build_dir,
        target_image=args.image,
        target=args.target,
        push=args.push,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        force_build=args.force_build,
        custom_tag_fn=custom_tag_fn,
    )


if __name__ == "__main__":
    sys.exit(main())
