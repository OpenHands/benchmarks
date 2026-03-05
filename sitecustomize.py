"""
Top-level sitecustomize to ensure our Modal logging patch is always applied.

Python will auto-import ``sitecustomize`` if it is importable on ``sys.path``.
During evaluation ``/workspace/benchmarks`` is on ``PYTHONPATH``, so placing
this file at the repo root guarantees the patch runs before swebench is used.
"""

import os
import sys


# ============================================================================
# CENTRALIZED LOGGING CONFIGURATION
# ============================================================================
# Disable rich logging to avoid threading issues with multiprocessing.
# Rich's RichHandler creates locks and threads that don't play well with fork().
os.environ["LOG_JSON"] = "1"

# Note: Verbose build stderr warnings (75k+ lines from uv build file copying)
# are suppressed in individual build_images.py files by setting the SDK build
# logger to ERROR level after SDK imports.

print("benchmarks sitecustomize imported", file=sys.stderr, flush=True)

try:
    # Reuse the actual patch logic that lives alongside the benchmarks package.
    from benchmarks.utils.sitecustomize import _apply_modal_logging_patch

    _apply_modal_logging_patch()
except Exception:
    # Avoid breaking startup for non-swebench runs; logging is best-effort.
    pass
