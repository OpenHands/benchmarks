"""
SWE-bench benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "princeton-nlp/SWE-bench_Verified",
    "split": "test",
    "workspace": "remote",
    "num_workers": 30,
    "max_iterations": 500,
    "max_attempts": 3,
    "max_retries": 3,
    "critic": "finish_with_patch",
    "output_dir": "./eval_outputs",
    "n_limit": 0,
    "note": "initial",
}

# Evaluation defaults (used by eval_infer.py)
EVAL_DEFAULTS = {
    "dataset": "princeton-nlp/SWE-bench_Verified",
    "model_name": "openhands",
    "workers": 12,
}
