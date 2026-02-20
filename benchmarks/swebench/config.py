"""
SWE-bench benchmark configuration.

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
INFER_DEFAULTS = {
    "dataset": "princeton-nlp/SWE-bench_Verified",
    "split": "test",
    "num_workers": 30,
    **CONDENSER_DEFAULTS,
}

# Evaluation defaults (used by eval_infer.py)
EVAL_DEFAULTS = {
    "dataset": "princeton-nlp/SWE-bench_Verified",
    "split": "test",
    "workers": 12,
    "modal": True,
    "timeout": 3600,
}

# Build defaults (used by build_images.py)
BUILD_DEFAULTS = {
    "max_workers": 32,
}
