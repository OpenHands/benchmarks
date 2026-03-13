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

print("benchmarks sitecustomize imported", file=sys.stderr, flush=True)

# Suppress verbose Docker build stderr warnings (75k+ lines from uv copying files).
# This MUST be done in sitecustomize.py to affect forked child processes.
# The SDK logs every "[stderr] copying..." line as WARNING, causing massive log output
# and 3x slowdown with JSON logging (Rich logging throttled these automatically).
try:
    import logging

    # Import SDK logger to trigger auto-configuration with LOG_JSON=1
    from openhands.sdk.logger import get_logger  # noqa: F401

    # Now suppress the verbose build logger in ALL processes (including forked children)
    logging.getLogger("openhands.agent_server.docker.build").setLevel(logging.ERROR)
except Exception:
    # Best-effort: don't break if SDK structure changes
    pass

try:
    # Reuse the actual patch logic that lives alongside the benchmarks package.
    from benchmarks.utils.sitecustomize import _apply_modal_logging_patch

    _apply_modal_logging_patch()
except Exception:
    # Avoid breaking startup for non-swebench runs; logging is best-effort.
    pass
