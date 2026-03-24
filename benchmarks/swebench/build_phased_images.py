#!/usr/bin/env python3
"""
Run the phased benchmark image build.

This is the dedicated entrypoint for the optimized path used by the workflows:
build the shared builder image, build the per-instance base images, then
assemble final images locally.
"""

import argparse
import sys

from benchmarks.swebench.build_base_images import (
    assemble_all_agent_images,
    build_all_base_images,
    build_builder_image,
)
from benchmarks.swebench.build_images import collect_unique_base_images, extract_custom_tag
from benchmarks.utils.build_utils import default_build_output_dir
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the phased benchmark image build."
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
    parser.add_argument(
        "--agent-type",
        type=str,
        default="default",
        help="Agent type: default, acp-claude, acp-codex",
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

    builder_result = build_builder_image(push=args.push)
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
    )
    if rc != 0:
        return rc

    def custom_tag_fn(base: str) -> str:
        tag = extract_custom_tag(base)
        if args.agent_type.startswith("acp-"):
            tag += "-acp"
        return tag

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
