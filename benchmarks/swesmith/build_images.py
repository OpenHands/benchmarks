#!/usr/bin/env python3
"""
Build agent-server images for all unique SWE-Smith base images in a dataset split.

Example:
  uv run benchmarks/swesmith/build_images.py \
    --dataset SWE-bench/SWE-smith-py --split train \
    --image ghcr.io/openhands/eval-agent-server --target source-minimal
"""

import sys

from benchmarks.utils.build_utils import (
    build_all_images,
    default_build_output_dir,
    get_build_parser,
)
from benchmarks.utils.dataset import get_dataset
from openhands.sdk import get_logger


logger = get_logger(__name__)


def get_official_docker_image(
    image_name: str,
) -> str:
    # Official SWE-Smith image - present already in HuggingFace dataset
    official_image_name: str = image_name.lower().strip()
    if not official_image_name.startswith("docker.io"):
        official_image_name = f"docker.io/{official_image_name}"
    logger.debug(f"Official SWE-Smith image: {official_image_name}")
    return official_image_name


def extract_custom_tag(base_image: str) -> str:
    """
    Extract SWE-Smith instance ID from official SWE-Smith image name.

    Example:
        docker.io/jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536
        -> swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536
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
    # discard all rows in dataset where problem_statement is empty
    df = df[df["problem_statement"].str.strip().astype(bool)]
    return sorted(
        {get_official_docker_image(str(row["image_name"])) for _, row in df.iterrows()}
    )


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
        post_build_fn=None,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
