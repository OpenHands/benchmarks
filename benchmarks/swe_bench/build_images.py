#!/usr/bin/env python3
"""
Build agent-server images for all unique SWE-Bench base images in a dataset split.

Example:
  uv run benchmarks/swe_bench/build_images.py \
    --dataset princeton-nlp/SWE-bench_Lite --split test \
    --image ghcr.io/all-hands-ai/agent-server --target binary-minimal
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.dataset import get_dataset
from openhands.agent_server.docker.build import BuildOptions, build
from openhands.sdk import get_logger


logger = get_logger(__name__)


def get_instance_docker_image(
    instance_id: str, prefix: str = "docker.io/swebench/"
) -> str:
    repo, name = instance_id.split("__")
    return f"{prefix.rstrip('/')}/sweb.eval.x86_64.{repo}_1776_{name}:latest".lower()


def extend_parser() -> argparse.ArgumentParser:
    """Reuse benchmark parser and extend with build-related options."""
    parser = get_parser(add_llm_config=False)
    parser.description = "Build all agent-server images for SWE-Bench base images."

    parser.add_argument(
        "--docker-image-prefix",
        default="docker.io/swebench/",
        help="Prefix for SWE-Bench images",
    )
    parser.add_argument(
        "--image",
        default="ghcr.io/all-hands-ai/agent-server",
        help="Target repo/name for built image",
    )
    parser.add_argument(
        "--target",
        default="binary-minimal",
        help="Build target (binary | binary-minimal | source | source-minimal)",
    )
    parser.add_argument(
        "--platforms", default="linux/amd64", help="Comma-separated platforms"
    )
    parser.add_argument("--custom-tags", default="", help="Comma-separated custom tags")
    parser.add_argument(
        "--push", action="store_true", help="Push via buildx instead of load locally"
    )
    parser.add_argument(
        "--max-workers", type=int, default=1, help="Concurrent builds (be cautious)"
    )
    parser.add_argument(
        "--manifest", default="build-manifest.json", help="Write build results here"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List base images only, don’t build"
    )
    return parser


def collect_unique_base_images(dataset, split, prefix, n_limit):
    df = get_dataset(
        dataset_name=dataset, split=split, eval_limit=n_limit if n_limit else None
    )
    return sorted(
        {
            get_instance_docker_image(str(row["instance_id"]), prefix)
            for _, row in df.iterrows()
        }
    )


def build_one(base_image, args):
    opts = BuildOptions(
        base_image=base_image,
        custom_tags=args.custom_tags,
        image=args.image,
        target=args.target,
        platforms=[p.strip() for p in args.platforms.split(",") if p.strip()],
        push=args.push,
    )
    tags = build(opts)
    return {"base_image": base_image, "tags": tags, "error": None}


def main(argv):
    parser = extend_parser()
    args = parser.parse_args(argv)

    bases = collect_unique_base_images(
        args.dataset, args.split, args.docker_image_prefix, args.n_limit
    )
    if args.dry_run:
        print("\n".join(bases))
        return 0

    results, failures = [], []
    if args.max_workers == 1:
        for base in bases:
            try:
                results.append(build_one(base, args))
            except Exception as e:
                logger.error("Build failed for %s: %r", base, e)
                failures.append({"base_image": base, "error": repr(e)})
                break
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {ex.submit(build_one, base, args): base for base in bases}
            for fut in as_completed(futures):
                base = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    logger.error("Build failed for %s: %r", base, e)
                    failures.append({"base_image": base, "error": repr(e)})
                    break

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "total_unique_base_images": len(bases),
        "built": len(results),
        "failed": len(failures),
        "results": results,
        "failures": failures,
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2))
    logger.info(
        "Done. Built=%d  Failed=%d  Manifest=%s",
        len(results),
        len(failures),
        args.manifest,
    )
    return 1 if failures and args.fail_fast else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
