"""
SWT-bench benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

# Condenser configuration
# The condenser manages conversation context by automatically truncating history
# when it exceeds max_size and replacing dropped events with an LLM-generated summary.
CONDENSER_DEFAULTS = {
    "enable_condenser": True,
    "condenser_max_size": 80,  # Maximum number of events before condensing
    "condenser_keep_first": 4,  # Number of initial events to always keep
}

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "eth-sri/SWT-bench_Verified_bm25_27k_zsp",
    "split": "test",
    "num_workers": 30,
    **CONDENSER_DEFAULTS,
}

# Evaluation defaults (used by eval_infer.py)
# Note: eval uses SWE-bench dataset, not SWT-bench dataset
EVAL_DEFAULTS = {
    "dataset": "princeton-nlp/SWE-bench_Verified",
    "split": "test",
    "workers": 24,
}

# Build defaults (used by build_images.py)
BUILD_DEFAULTS = {
    "max_workers": 16,
}
