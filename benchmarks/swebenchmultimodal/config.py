"""
SWE-bench Multimodal benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "princeton-nlp/SWE-bench_Multimodal",
    "split": "dev",
    "workspace": "remote",
    "num_workers": 30,
    "max_iterations": 500,
    "max_attempts": 3,
    "max_retries": 3,
    "critic": "finish_with_patch",
}

# Evaluation defaults (used by eval_infer.py)
EVAL_DEFAULTS = {
    "dataset": "princeton-nlp/SWE-bench_Multimodal",
    "split": "dev",
    "model_name": "openhands",
    "workers": 12,
}
