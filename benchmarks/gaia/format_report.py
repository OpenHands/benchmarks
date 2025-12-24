#!/usr/bin/env python3
"""
GAIA Report Formatter

Delegates to the SWE-bench formatter while customizing the header.

Usage:
    python format_report.py <output.jsonl> <report.json> [--env-file <env_file>]
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if "--benchmark-name" not in sys.argv:
        sys.argv.extend(["--benchmark-name", "GAIA"])
    swebench_path = Path(__file__).resolve().parents[1] / "swebench" / "format_report.py"
    runpy.run_path(str(swebench_path), run_name="__main__")


if __name__ == "__main__":
    main()
