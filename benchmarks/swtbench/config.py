"""
SWT-bench benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "eth-sri/SWT-bench_Verified_bm25_27k_zsp",
    "split": "test",
    "workspace": "remote",
    "num_workers": 30,
    "max_iterations": 500,
    "max_attempts": 3,
    "max_retries": 3,
    "critic": "finish_with_patch",
}

# Evaluation defaults (used by eval_infer.py)
# Note: eval uses SWE-bench dataset, not SWT-bench dataset
EVAL_DEFAULTS = {
    "dataset": "princeton-nlp/SWE-bench_Verified",
    "model_name": "OpenHands",
    "workers": 24,
}
