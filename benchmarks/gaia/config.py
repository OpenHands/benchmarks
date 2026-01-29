"""
GAIA benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "gaia-benchmark/GAIA",
    "split": "validation",
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
    "model_name": "openhands",
    "workers": 1,
}
