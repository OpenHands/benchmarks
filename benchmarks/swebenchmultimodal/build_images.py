#!/usr/bin/env python3
"""
Build agent-server images for all unique SWE-Bench Multimodal base images in a dataset split.

Uses a three-phase build pipeline (shared with SWE-bench):
  1. Build shared builder image (SDK + dependencies)
  2. Build per-instance base images
  3. Assemble final agent images locally

Example:
  uv run benchmarks/swebenchmultimodal/build_images.py \
    --dataset princeton-nlp/SWE-bench_Multimodal --split dev \
    --image ghcr.io/openhands/eval-agent-server
"""

import argparse
import sys

from benchmarks.swebenchmultimodal.config import BUILD_DEFAULTS
from benchmarks.utils.build_utils import default_build_output_dir
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.dataset import get_dataset
from openhands.sdk import get_logger


logger = get_logger(__name__)


def get_official_docker_image(
    instance_id: str,
    docker_image_prefix="docker.io/swebench/",
) -> str:
    # For multimodal benchmark, we use regular SWE-bench images as base
    # since multimodal-specific images (sweb.mm.eval.*) are not available
    # The multimodal functionality is handled at the application level
    repo, name = instance_id.split("__")

    # Use regular SWE-bench image as base for multimodal instances
    regular_image_name = docker_image_prefix.rstrip("/")
    regular_image_name += f"/sweb.eval.x86_64.{repo}_1776_{name}:latest".lower()

    logger.debug(
        f"Using regular SWE-Bench image for multimodal instance: {regular_image_name}"
    )
    return regular_image_name


def extract_custom_tag(base_image: str) -> str:
    """
    Extract instance ID from official SWE-Bench image name (multimodal or regular).

    Examples:
        docker.io/swebench/sweb.mm.eval.x86_64.openlayers_1776_openlayers-12172:latest
        -> sweb.mm.eval.x86_64.openlayers_1776_openlayers-12172

        docker.io/swebench/sweb.eval.x86_64.django_1776_django-11333:latest
        -> sweb.eval.x86_64.django_1776_django-11333
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


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build agent-server images for SWE-Bench Multimodal using the three-phase pipeline."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=BUILD_DEFAULTS.get("dataset", "princeton-nlp/SWE-bench_Multimodal"),
        help="Dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=BUILD_DEFAULTS.get("split", "dev"),
        help="Dataset split",
    )
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
        default=BUILD_DEFAULTS.get("max_workers", 12),
        help="Concurrent builds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
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
    # The three-phase build functions live in benchmarks.swebench.build_base_images
    # because the infrastructure is identical (same SDK Dockerfile, same base images,
    # same builder/base/assembly pattern). Multimodal images differ only at inference
    # time, not at the Docker image level.
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

    # Phase 0: Build shared builder image (SDK + dependencies)
    builder_result = build_builder_image(push=args.push)
    if builder_result.error or not builder_result.tags:
        print(
            builder_result.error or "Builder image build produced no tags",
            file=sys.stderr,
        )
        return 1

    # Phase 1: Build per-instance base images
    rc = build_all_base_images(
        base_images=base_images,
        build_dir=build_dir,
        push=args.push,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )
    if rc != 0:
        return rc

    # Phase 2: Assemble final agent images locally
    # No wrapping needed for multimodal (no docutils/roman dependency)
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
