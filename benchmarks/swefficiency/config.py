"""
SWE-fficiency benchmark configuration.

Default values for the SWE-fficiency performance optimization benchmark.
"""

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "swefficiency/swefficiency",
    "split": "test",
    "num_workers": 4,
}

# Docker resource defaults
DOCKER_DEFAULTS = {
    "num_cpus_per_worker": 4,
    "mem_limit": "16g",
    "num_cpus_to_skip": 0,
}
