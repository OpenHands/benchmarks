"""
SWE-Bench hyperparameters and constant values.

This module serves as the single source of truth for all constant values
used in the SWE-Bench evaluation workflow.
"""

# =============================================================================
# Dataset Configuration
# =============================================================================
DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"

# =============================================================================
# Docker Image Configuration
# =============================================================================
DOCKER_IMAGE_PREFIX = "docker.io/swebench/"
DOCKER_IMAGE_TAG = "latest"

# =============================================================================
# Build Configuration
# =============================================================================
BUILD_TARGET_SOURCE_MINIMAL = "source-minimal"
BUILD_TARGET_BINARY = "binary"
DEFAULT_BUILD_TARGET = BUILD_TARGET_SOURCE_MINIMAL

# =============================================================================
# Runtime Configuration
# =============================================================================
DEFAULT_RUNTIME_API_URL = "https://runtime.eval.all-hands.dev"
DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT = "600"

# =============================================================================
# Evaluation Configuration
# =============================================================================
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_NUM_WORKERS = 1
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_RETRIES = 3
DEFAULT_EVAL_WORKERS = "12"
DEFAULT_N_LIMIT = 0
DEFAULT_NOTE = "initial"
DEFAULT_OUTPUT_DIR = "./eval_outputs"

# =============================================================================
# Model Configuration
# =============================================================================
DEFAULT_MODEL_NAME = "openhands"

# =============================================================================
# Git Configuration
# =============================================================================
GIT_USER_EMAIL = "evaluation@openhands.dev"
GIT_USER_NAME = "OpenHands Evaluation"
GIT_COMMIT_MESSAGE = "patch"

# =============================================================================
# Patch Processing
# =============================================================================
SETUP_FILES_TO_REMOVE = ["pyproject.toml", "tox.ini", "setup.py"]
