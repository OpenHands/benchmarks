"""
Sitecustomize injected into the Modal function image for SWE-bench runs.

This file is copied into the Modal function container and imported automatically
by Python (via sitecustomize) to patch the modal_eval runtime with extra timing
logs and a libmamba solver setup.
"""

from __future__ import annotations

import sys


def _apply_modal_image_patch() -> None:
    try:
        from benchmarks.utils import modal_patches
    except Exception:
        try:
            import modal_patches
        except Exception as exc:
            print(
                f"[benchmarks] modal sitecustomize: failed to import modal_patches: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return
    modal_patches.apply_image_patches()


_apply_modal_image_patch()
