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
    "mem_limit": "32g",
    "num_cpus_to_skip": 4,  # Skip first N CPUs to avoid contention with host processes
    "cleanup_agent_image": True,
    "cleanup_base_image": True,
    "prune_buildkit_cache": False,
}
