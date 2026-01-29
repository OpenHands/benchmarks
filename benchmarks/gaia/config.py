"""
GAIA benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "gaia-benchmark/GAIA",
    "split": "validation",
    "num_workers": 30,
    "max_attempts": 3,
}
