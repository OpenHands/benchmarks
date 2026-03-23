#!/usr/bin/env python3
"""
SWT-Bench image build shim.

SWT-Bench uses the same base environment images and build flow as SWE-Bench.
This module simply forwards to the SWE-Bench build logic to avoid duplication
while keeping the SWT entrypoint stable for workflows.

Note: SWT-bench uses max_workers=16 (vs SWE-bench's 32) via BUILD_DEFAULTS.
"""

import sys

from benchmarks.swebench.build_images import (
    _wrap_if_needed,
    collect_unique_base_images,
    extract_custom_tag,
)
from benchmarks.swtbench.config import BUILD_DEFAULTS
from benchmarks.utils.build_utils import (
    build_all_images,
    build_args_for_agent_type,
    default_build_output_dir,
    get_build_parser,
)
from openhands.sdk import get_logger


logger = get_logger(__name__)


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

    # Append ACP suffix to image tags so ACP and non-ACP images are distinct.
    # This prevents IfNotPresent pull policy from using stale non-ACP images.
    def tag_fn(base: str) -> str:
        tag = extract_custom_tag(base)
        if args.agent_type.startswith("acp-"):
            tag += "-acp"
        return tag

    return build_all_images(
        base_images=base_images,
        target=args.target,
        build_dir=build_dir,
        image=args.image,
        push=args.push,
        max_workers=args.max_workers,
        build_batch_size=args.build_batch_size,
        dry_run=args.dry_run,
        force_build=args.force_build,
        max_retries=args.max_retries,
        base_image_to_custom_tag_fn=tag_fn,
        post_build_fn=_wrap_if_needed,
        extra_build_args=build_args_for_agent_type(args.agent_type),
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
