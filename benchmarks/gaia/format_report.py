#!/usr/bin/env python3
"""
GAIA Report Formatter

Delegates to the SWE-bench formatter while customizing the header.

Usage:
    python format_report.py <output.jsonl> <report.json> [--env-file <env_file>]
"""

from __future__ import annotations

import sys

from benchmarks.swebench import format_report as swebench_format_report


def main() -> None:
    if "--benchmark-name" not in sys.argv:
        sys.argv.extend(["--benchmark-name", "GAIA"])
    swebench_format_report.main()


if __name__ == "__main__":
    main()
