"""
SWE-bench benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

from benchmarks.swebench.constants import (
    DEFAULT_CLI_MODEL_NAME,
    DEFAULT_DATASET,
    DEFAULT_EVAL_WORKERS,
)


# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": DEFAULT_DATASET,
    "split": "test",
    "workspace": "remote",
    "num_workers": 30,
    "max_iterations": 500,
    "max_attempts": 3,
    "max_retries": 3,
    "critic": "finish_with_patch",
}

# Evaluation defaults (used by eval_infer.py)
EVAL_DEFAULTS = {
    "dataset": DEFAULT_DATASET,
    "model_name": DEFAULT_CLI_MODEL_NAME,
    "workers": DEFAULT_EVAL_WORKERS,
}
