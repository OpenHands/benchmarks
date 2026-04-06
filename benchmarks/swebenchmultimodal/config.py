"""
SWE-bench Multimodal benchmark configuration.

Default values aligned with evaluation repository (OpenHands/evaluation).
"""

from pathlib import Path


DEFAULT_RESOLVED_INSTANCES_FILE = Path(__file__).with_name("resolved_instances.txt")

# Condenser configuration
# The condenser manages conversation context by automatically truncating history
# when it exceeds max_size and replacing dropped events with an LLM-generated summary.
CONDENSER_DEFAULTS = {
    "enable_condenser": True,
    "condenser_max_size": 240,  # Maximum number of events before condensing
    "condenser_keep_first": 2,  # Number of initial events to always keep
}

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "princeton-nlp/SWE-bench_Multimodal",
    "split": "dev",
    "num_workers": 30,
    "select": str(DEFAULT_RESOLVED_INSTANCES_FILE),
    **CONDENSER_DEFAULTS,
}

# Evaluation defaults (used by eval_infer.py)
EVAL_DEFAULTS = {
    "dataset": "princeton-nlp/SWE-bench_Multimodal",
    "split": "dev",
    "workers": 12,
    "modal": True,
}

# Build defaults (used by build_images.py)
BUILD_DEFAULTS = {
    "max_workers": 32,
}
