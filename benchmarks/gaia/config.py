"""
GAIA benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "gaia-benchmark/GAIA",
    "split": "validation",
    "level": "2023_all",
    "num_workers": 30,
}

# Build defaults (used by build_images.py)
BUILD_DEFAULTS = {
    "max_workers": 1,
}
