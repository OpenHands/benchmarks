#!/usr/bin/env python3
"""Build agent-server images for SWE-Bench Pro instances."""

import argparse
import sys
from collections.abc import Mapping
from typing import Any

from benchmarks.swebench.build_base_images import (
    assemble_all_agent_images,
    build_all_base_images,
    build_builder_image,
)
from benchmarks.swebenchpro import constants
from benchmarks.utils.build_utils import default_build_output_dir
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.dataset import get_dataset
from openhands.sdk import get_logger


logger = get_logger(__name__)


def get_official_docker_image(
    instance: Mapping[str, Any],
    docker_image_prefix: str = constants.DOCKER_IMAGE_PREFIX,
) -> str:
    dockerhub_tag = str(instance.get("dockerhub_tag", "")).strip()
    if not dockerhub_tag:
        raise ValueError(
            f"Missing dockerhub_tag for instance {instance.get('instance_id')}"
        )
    return f"{docker_image_prefix}:{dockerhub_tag}"


def extract_custom_tag(base_image: str) -> str:
    _, _, tag = base_image.rpartition(":")
    if not tag:
        raise ValueError(f"Could not extract docker tag from image: {base_image}")
    return tag


def collect_unique_base_images(
    dataset: str,
    split: str,
    n_limit: int,
    selected_instances_file: str | None = None,
) -> list[str]:
    df = get_dataset(
        dataset_name=dataset,
        split=split,
        eval_limit=n_limit if n_limit else None,
        selected_instances_file=selected_instances_file,
    )
    return sorted(
        {get_official_docker_image(row.to_dict()) for _, row in df.iterrows()}
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build agent-server images for SWE-Bench Pro instances."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ScaleAI/SWE-bench_Pro",
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
        custom_tag_fn=extract_custom_tag,
    )
    if rc != 0:
        return rc

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
        custom_tag_fn=extract_custom_tag,
    )


if __name__ == "__main__":
    sys.exit(main())
