"""
Site-wide hooks for benchmarks.

When running SWE-Bench evaluation on Modal, we want to capture exceptions that
happen before a `report.json` is written (e.g., sandbox creation failures). The
upstream harness only prints these exceptions, so the scoring step sees missing
logs and marks the instance as a generic error. This module installs patches to
persist a minimal log/report for any exception result.

We also patch the scikit-learn install command used inside Modal sandboxes to
drop the deprecated `--no-use-pep517` flag (removed in pip>=25). That flag
breaks the sandbox image build before any logs are produced.

This file is imported automatically by Python when present on `sys.path`
(`PYTHONPATH` already includes `/workspace/benchmarks` in the evaluation job),
so no extra wiring is needed.
"""

from __future__ import annotations

from benchmarks.utils import modal_patches


def _apply_modal_logging_patch() -> None:
    modal_patches.apply_host_patches()


_apply_modal_logging_patch()
