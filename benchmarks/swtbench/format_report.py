#!/usr/bin/env python3
"""
SWT-Bench Report Formatter

Delegates formatting to the SWE-bench formatter since the report schema matches.
"""

import sys

from benchmarks.swebench import format_report as swebench_format_report


def main() -> None:
    if "--benchmark-name" not in sys.argv:
        sys.argv.extend(["--benchmark-name", "SWT-Bench"])
    swebench_format_report.main()


if __name__ == "__main__":
    main()
