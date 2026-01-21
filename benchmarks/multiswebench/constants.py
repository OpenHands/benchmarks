"""
Multi-SWE-bench benchmark-specific constants.

This file contains constant values specific to the Multi-SWE-bench benchmark.
Shared constants are defined in benchmarks/utils/constants.py.
"""

import os
from pathlib import Path


# Dataset configuration
MULTISWEBENCH_DATASET = "ByteDance-Seed/Multi-SWE-bench"
MULTISWEBENCH_DEFAULT_SPLIT = "test"
DEFAULT_LANG = "java"

# Docker image configuration
DEFAULT_DOCKER_IMAGE_PREFIX = "mswebench"
DOCKER_IMAGE_PREFIX = os.environ.get(
    "EVAL_DOCKER_IMAGE_PREFIX", DEFAULT_DOCKER_IMAGE_PREFIX
)

# Cache directory for Multi-SWE-bench dataset files
DATASET_CACHE_DIR = Path(__file__).parent / "data"

# Build target configuration
DEFAULT_BUILD_TARGET = "source-minimal"

# Environment variables for Multi-SWE-Bench configuration
USE_HINT_TEXT = os.environ.get("USE_HINT_TEXT", "false").lower() == "true"
USE_INSTANCE_IMAGE = os.environ.get("USE_INSTANCE_IMAGE", "true").lower() == "true"
RUN_WITH_BROWSING = os.environ.get("RUN_WITH_BROWSING", "false").lower() == "true"
