"""
SWE-bench benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

# Condenser configuration
# The condenser manages conversation context by automatically truncating history
# when it exceeds max_size and replacing dropped events with an LLM-generated summary.
CONDENSER_DEFAULTS = {
    "enable_condenser": True,
    "condenser_max_size": 80,  # Temporary eval override for GPT-5.4 50-instance run
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
