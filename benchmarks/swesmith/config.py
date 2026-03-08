"""
SWE-Smith benchmark configuration.
"""

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "SWE-bench/SWE-smith-py",
    "split": "train",
    "num_workers": 4,
}

# Evaluation defaults (used by eval_infer.py)
EVAL_DEFAULTS = {
    "workers": 4,
}
