#!/usr/bin/env python3
"""
Wrapper script to build agent-server images for SWT-Bench.

SWT-Bench uses the same image build flow as SWE-Bench; we reuse that
implementation here so the build workflow has a stable entrypoint.
"""

import sys

from benchmarks.swebench.build_images import main as swebench_build_main


def main(argv: list[str]) -> int:
    return swebench_build_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
