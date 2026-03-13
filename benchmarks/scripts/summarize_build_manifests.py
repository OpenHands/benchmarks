from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks.utils.build_manifest import (
    render_build_summary_markdown,
    summarize_build_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a markdown or JSON summary from build manifest files."
    )
    parser.add_argument(
        "--build-root",
        default="builds",
        help="Root directory containing manifest.jsonl build artifacts",
    )
    parser.add_argument(
        "--title",
        default="Build Summary",
        help="Title to use in markdown output",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of slowest built images to include",
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit non-zero if failed builds are present",
    )
    args = parser.parse_args()

    summary = summarize_build_root(Path(args.build_root), top_n=args.top_n)
    if args.format == "json":
        print(summary.model_dump_json(indent=2))
    else:
        print(render_build_summary_markdown(summary, title=args.title))

    return 1 if args.fail_on_errors and summary.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
