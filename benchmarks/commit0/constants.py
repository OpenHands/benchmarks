"""
Commit0 benchmark-specific constants.

This file contains constant values specific to the Commit0 benchmark.
Shared constants are defined in benchmarks/utils/constants.py.
"""

import os


# Dataset configuration
COMMIT0_DATASET = "wentingzhao/commit0_combined"
COMMIT0_DEFAULT_SPLIT = "test"
DEFAULT_REPO_SPLIT = "lite"

# Docker image configuration
DEFAULT_DOCKER_IMAGE_PREFIX = "docker.io/wentingzhao/"
DOCKER_IMAGE_PREFIX = os.environ.get(
    "EVAL_DOCKER_IMAGE_PREFIX", DEFAULT_DOCKER_IMAGE_PREFIX
)

# Build target configuration
DEFAULT_BUILD_TARGET = "source-minimal"
