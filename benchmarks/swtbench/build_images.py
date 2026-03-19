#!/usr/bin/env python3
"""
SWT-Bench image build shim.

SWT-Bench uses the same base environment images and build flow as SWE-Bench.
This module simply forwards to the SWE-Bench build logic to avoid duplication
while keeping the SWT entrypoint stable for workflows.

Note: SWT-bench uses max_workers=16 (vs SWE-bench's 32) via BUILD_DEFAULTS.
"""

import sys

from benchmarks.swebench import constants as swebench_constants
from benchmarks.swebench.build_images import (
    _make_agent_layer_build_fn,
    _wrap_if_needed,
    collect_unique_base_images,
    eval_base_tag_for_custom_tag,
    extract_custom_tag,
)
from benchmarks.swtbench.config import BUILD_DEFAULTS
from benchmarks.utils.build_utils import (
    build_all_base_images,
    build_all_images,
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

    if args.dry_run:
        print("\n".join(base_images))
        return 0

    # ── Phase 1: build eval-base images ──
    logger.info(
        "Phase 1: building %d eval-base images (SDK-independent)", len(base_images)
    )
    base_rc = build_all_base_images(
        base_images=base_images,
        base_image_to_custom_tag_fn=extract_custom_tag,
        base_image_tag_version=swebench_constants.BASE_IMAGE_TAG_VERSION,
        build_dir=build_dir,
        push=args.push,
        max_workers=args.max_workers,
        force_build=args.force_build,
        max_retries=args.max_retries,
    )
    if base_rc != 0:
        logger.error("Phase 1 failed: some base images could not be built.")
        return base_rc

    # ── Phase 2: build thin agent layers ──
    logger.info("Phase 2: building %d agent layers on cached bases", len(base_images))
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
