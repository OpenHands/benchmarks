"""
SWE-Bench hyperparameters and constant values.

This module serves as the single source of truth for all constant values
used in the SWE-Bench evaluation workflow.
"""

from typing import Final


# Dataset
DEFAULT_DATASET: Final[str] = "princeton-nlp/SWE-bench_Verified"

# Docker
DOCKER_IMAGE_PREFIX: Final[str] = "docker.io/swebench/"
DOCKER_IMAGE_TAG: Final[str] = "latest"
WRAPPED_REPOS: Final[frozenset[str]] = frozenset(
    {"sphinx-doc"}
)  # Repos requiring docutils/roman wrapper

# Build
BUILD_TARGET_SOURCE_MINIMAL: Final[str] = "source-minimal"
BUILD_TARGET_BINARY: Final[str] = "binary"
DEFAULT_BUILD_TARGET: Final[str] = BUILD_TARGET_SOURCE_MINIMAL

# Runtime
DEFAULT_RUNTIME_API_URL: Final[str] = "https://runtime.eval.all-hands.dev"
DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT: Final[int] = 600

# Evaluation
DEFAULT_EVAL_WORKERS: Final[int] = 12

# Model - preserving original behavior: function default is "OpenHands", CLI default is "openhands"
DEFAULT_MODEL_NAME: Final[str] = "OpenHands"
DEFAULT_CLI_MODEL_NAME: Final[str] = "openhands"

# Git
GIT_USER_EMAIL: Final[str] = "evaluation@openhands.dev"
GIT_USER_NAME: Final[str] = "OpenHands Evaluation"
GIT_COMMIT_MESSAGE: Final[str] = "patch"

# Patch Processing
SETUP_FILES_TO_REMOVE: Final[tuple[str, ...]] = (
    "pyproject.toml",
    "tox.ini",
    "setup.py",
)
