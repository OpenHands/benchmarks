"""
SWT-bench benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "eth-sri/SWT-bench_Verified_bm25_27k_zsp",
    "split": "test",
    "num_workers": 30,
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
