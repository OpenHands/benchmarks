"""
Commit0 Benchmark Constants

This module serves as the single source of truth for all hyperparameters
and constant values used in the Commit0 benchmark evaluation workflow.
"""

# Dataset configuration
DEFAULT_DATASET = "wentingzhao/commit0_combined"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_REPO_SPLIT = "lite"

# Docker image configuration
DEFAULT_DOCKER_IMAGE_PREFIX = "docker.io/wentingzhao/"
DEFAULT_IMAGE_TAG = "v0"
CUSTOM_TAG_PREFIX = "commit0-"

# Build configuration
BUILD_TARGET = "source-minimal"

# Workspace configuration
WORKSPACE_DIR = "/workspace"

# Git configuration
GIT_BRANCH_NAME = "commit0_combined"
AGENT_BRANCH_NAME = "openhands"

# Model configuration
DEFAULT_MODEL_NAME = "openhands"

# Runtime configuration
DEFAULT_RUNTIME_API_URL = "https://runtime.eval.all-hands.dev"
DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT = 600
DEFAULT_CONVERSATION_TIMEOUT = 3600
DEFAULT_COMMAND_TIMEOUT = 600

# Evaluation configuration
TOTAL_INSTANCES = 16
