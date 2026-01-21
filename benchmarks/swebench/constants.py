"""
SWE-bench benchmark-specific constants.

This file contains constant values specific to the SWE-bench benchmark.
Shared constants are defined in benchmarks/utils/constants.py.
"""

# Dataset configuration
SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Verified"
SWEBENCH_DEFAULT_SPLIT = "test"

# Docker image configuration
SWEBENCH_DOCKER_IMAGE_PREFIX = "docker.io/swebench/"

# Repos that require the docutils/roman wrapper layer
WRAPPED_REPOS = {"sphinx-doc"}

# Build target configuration
DEFAULT_BUILD_TARGET = "source-minimal"
