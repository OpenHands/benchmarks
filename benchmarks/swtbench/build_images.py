#!/usr/bin/env python3
"""
SWT-Bench image build shim.

SWT-Bench uses the same base environment images and build flow as SWE-Bench.
This module simply forwards to the SWE-Bench build logic to avoid duplication
while keeping the SWT entrypoint stable for workflows.
"""

import sys

from benchmarks.swebench.build_images import (
    main as swebench_main,
)


# Re-export the SWE-Bench logic under the SWT entrypoint
def main(argv: list[str]) -> int:  # pragma: no cover - thin wrapper
    return swebench_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
