#!/usr/bin/env python3
"""
Build agent-server images for all unique SWT-Bench base images in a dataset split.

SWT-Bench uses the same base environment images as SWE-Bench (from docker.io/swebench/),
but the task is different - generating reproduction tests instead of patches.

Example:
  uv run benchmarks/swtbench/build_images.py \
    --dataset princeton-nlp/SWE-bench_Verified --split test \
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
    instance_id: str,
    docker_image_prefix="docker.io/swebench/",
) -> str:
    # Official SWT-Bench uses same base images as SWE-Bench
    # swebench/sweb.eval.x86_64.django_1776_django-11333:latest
    repo, name = instance_id.split("__")
    official_image_name = docker_image_prefix.rstrip("/")
    official_image_name += f"/sweb.eval.x86_64.{repo}_1776_{name}:latest".lower()
    logger.debug(f"Official SWT-Bench image: {official_image_name}")
    return official_image_name


def extract_custom_tag(base_image: str) -> str:
    """
    Extract SWT-Bench instance ID from official SWT-Bench image name.

    Example:
        docker.io/swtbench/sweb.eval.x86_64.django_1776_django-12155:latest
        -> sweb.eval.x86_64.django_1776_django-12155
    """
    name_tag = base_image.split("/")[-1]
    name = name_tag.split(":")[0]
    return name


def collect_unique_base_images(dataset, split, n_limit):
    df = get_dataset(
        dataset_name=dataset, split=split, eval_limit=n_limit if n_limit else None
    )
    return sorted(
        {get_official_docker_image(str(row["instance_id"])) for _, row in df.iterrows()}
    )


def main(argv: list[str]) -> int:
    parser = get_build_parser()
    args = parser.parse_args(argv)

    base_images: list[str] = collect_unique_base_images(
        args.dataset, args.split, args.n_limit
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
