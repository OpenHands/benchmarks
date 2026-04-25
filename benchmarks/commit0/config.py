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

# Timeout for each conversation run in seconds.
# commit0 instances are large repos that need extended time:
# marshmallow finished in ~98 min; babel/chardet need >120 min — 10800s (3h).
CONVERSATION_TIMEOUT = 10800

# Inference defaults (used by run_infer.py)
# Note: commit0 uses n_critic_runs=1 and max_retries=1 (different from default of 3)
INFER_DEFAULTS = {
    "dataset": "wentingzhao/commit0_combined",
    "split": "test",
    "repo_split": "lite",
    "num_workers": 16,
    "n_critic_runs": 1,
    "max_retries": 3,
    **CONDENSER_DEFAULTS,
}

# Commit0 needs the source-mode runtime, but it cannot use the SDK's direct
# source-minimal build path because the resulting venv points at a system Python
# path that does not exist in the upstream commit0 base images.
#
# The benchmark-side fix is to keep using source-minimal while routing commit0
# through the phased assembly path in benchmarks, which copies the runtime into
# the final image with the wrapper Dockerfile.
BUILD_TARGET = "source-minimal"

# Build defaults (used by build_images.py)
BUILD_DEFAULTS = {
    "max_workers": 16,
    "target": BUILD_TARGET,
}
