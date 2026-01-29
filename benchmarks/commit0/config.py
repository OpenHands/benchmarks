"""
Commit0 benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "wentingzhao/commit0_combined",
    "split": "test",
    "repo_split": "lite",
    "num_workers": 8,
    "max_iterations": 500,
    "max_attempts": 1,
    "max_retries": 1,
    "critic": "finish_with_patch",
}
