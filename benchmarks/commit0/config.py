"""
Commit0 benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

# Condenser configuration
# The condenser manages conversation context by automatically truncating history
# when it exceeds max_size and replacing dropped events with an LLM-generated summary.
CONDENSER_DEFAULTS = {
    "enable_condenser": True,
    "condenser_max_size": 240,  # Maximum number of events before condensing
    "condenser_keep_first": 2,  # Number of initial events to always keep
}

# Inference defaults (used by run_infer.py)
# Note: commit0 uses max_attempts=1 and max_retries=1 (different from default of 3)
INFER_DEFAULTS = {
    "dataset": "wentingzhao/commit0_combined",
    "split": "test",
    "repo_split": "lite",
    "num_workers": 16,
    "max_attempts": 1,
    "max_retries": 3,
    **CONDENSER_DEFAULTS,
}

# Build defaults (used by build_images.py)
BUILD_DEFAULTS = {
    "max_workers": 16,
}
