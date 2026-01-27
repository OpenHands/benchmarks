"""
GAIA benchmark constants and hyperparameters.

This module serves as the single source of truth for all constant values
used throughout the GAIA benchmark implementation.
"""

from typing import Final, Literal


# =============================================================================
# Dataset Configuration
# =============================================================================
DATASET_NAME: Final[str] = "gaia-benchmark/GAIA"
DATASET_YEAR: Final[str] = "2023"
DATASET_SPLIT_VALIDATION: Final[str] = "validation"
DATASET_SPLIT_TEST: Final[str] = "test"
GAIA_LEVELS: Final[list[str]] = [
    "2023_level1",
    "2023_level2",
    "2023_level3",
    "2023_all",
]

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
DEFAULT_MAX_ITERATIONS: Final[int] = 30
DEFAULT_CRITIC: Final[str] = "pass"
DEFAULT_OUTPUT_DIR: Final[str] = "outputs"
DEFAULT_NUM_WORKERS: Final[int] = 1
DEFAULT_MAX_ATTEMPTS: Final[int] = 1
