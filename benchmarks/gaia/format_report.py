#!/usr/bin/env python3
"""
GAIA Report Formatter

Delegates to the SWE-bench formatter while customizing the header.

Usage:
    python format_report.py <output.jsonl> <report.json> [--env-file <env_file>]
"""

from __future__ import annotations

import os

from benchmarks.swebench import format_report as swebench_format_report


def main() -> None:
    os.environ.setdefault("BENCHMARK_DISPLAY_NAME", "GAIA")
    swebench_format_report.main()


if __name__ == "__main__":
    main()
