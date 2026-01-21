"""
GAIA benchmark-specific constants.

This file contains constant values specific to the GAIA benchmark.
Shared constants are defined in benchmarks/utils/constants.py.
"""

from pathlib import Path


# Dataset configuration
GAIA_DATASET = "gaia-benchmark/GAIA"
GAIA_DEFAULT_SPLIT = "validation"

# Docker image configuration
GAIA_BASE_IMAGE = "nikolaik/python-nodejs:python3.12-nodejs22"

# Cache directory for GAIA dataset files
DATASET_CACHE_DIR = Path(__file__).parent / "data"

# GAIA data year (used for file paths)
GAIA_DATA_YEAR = "2023"
