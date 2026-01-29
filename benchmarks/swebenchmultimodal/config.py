"""
SWE-bench Multimodal benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "princeton-nlp/SWE-bench_Multimodal",
    "split": "dev",
    "num_workers": 30,
}

# Evaluation defaults (used by eval_infer.py)
EVAL_DEFAULTS = {
    "dataset": "princeton-nlp/SWE-bench_Multimodal",
    "split": "dev",
    "workers": 12,
}

# Build defaults (used by build_images.py)
BUILD_DEFAULTS = {
    "max_workers": 32,
}
