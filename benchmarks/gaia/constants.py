"""
GAIA benchmark constants and hyperparameters.

This module serves as the single source of truth for all constant values
used throughout the GAIA benchmark implementation.

Note: Default values for CLI arguments (max_iterations, num_workers, output_dir,
max_attempts, critic, split) are defined in benchmarks/utils/args_parser.py
which is the shared argument parser used by all benchmarks.
"""

from typing import Final, Literal


# =============================================================================
# Dataset Configuration
# =============================================================================
DATASET_NAME: Final[str] = "gaia-benchmark/GAIA"
DATASET_YEAR: Final[str] = "2023"
DATASET_SPLIT_VALIDATION: Final[str] = "validation"

# =============================================================================
# Docker/Image Configuration
# =============================================================================
GAIA_BASE_IMAGE: Final[str] = "nikolaik/python-nodejs:python3.12-nodejs22"
TARGET_TYPE: Final[Literal["binary", "source"]] = "binary"
IMAGE_TAG_PREFIX: Final[str] = "gaia"

# =============================================================================
# Runtime Configuration
# =============================================================================
DEFAULT_RUNTIME_API_URL: Final[str] = "https://runtime.eval.all-hands.dev"
DEFAULT_STARTUP_TIMEOUT: Final[float] = 600.0

# =============================================================================
# Default Values
# =============================================================================
DEFAULT_MODEL_NAME: Final[str] = "openhands"
