"""
Commit0 benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

# Inference defaults (used by run_infer.py)
# Note: commit0 uses max_attempts=1 and max_retries=1 (different from default of 3)
INFER_DEFAULTS = {
    "dataset": "wentingzhao/commit0_combined",
    "split": "test",
    "repo_split": "lite",
    "num_workers": 8,
    "max_attempts": 1,
    "max_retries": 1,
}

# Build defaults (used by build_images.py)
BUILD_DEFAULTS = {
    "max_workers": 16,
}
