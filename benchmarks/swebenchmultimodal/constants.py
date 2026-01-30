"""
Constants for SWE-Bench Multimodal benchmark.

This module serves as the single source of truth for all hyperparameters
and constant values used in the SWE-Bench Multimodal evaluation workflow.
"""

# Dataset configuration
DEFAULT_DATASET = "princeton-nlp/SWE-bench_Multimodal"
DEFAULT_SPLIT = "dev"

# Docker image configuration
DOCKER_IMAGE_PREFIX = "docker.io/swebench/"

# Build configuration
BUILD_TARGET = "source-minimal"

# Workspace configuration
WORKSPACE_DIR = "/workspace"

# Environment variable names
ENV_SKIP_BUILD = "SKIP_BUILD"
ENV_RUNTIME_API_KEY = "RUNTIME_API_KEY"
ENV_SDK_SHORT_SHA = "SDK_SHORT_SHA"
ENV_REMOTE_RUNTIME_STARTUP_TIMEOUT = "REMOTE_RUNTIME_STARTUP_TIMEOUT"
ENV_RUNTIME_API_URL = "RUNTIME_API_URL"

# Default values for environment variables
DEFAULT_SKIP_BUILD = "1"
DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT = "600"
DEFAULT_RUNTIME_API_URL = "https://runtime.eval.all-hands.dev"

# Git configuration
GIT_USER_EMAIL = "evaluation@openhands.dev"
GIT_USER_NAME = "OpenHands Evaluation"
GIT_COMMIT_MESSAGE = "patch"

# Environment setup commands
ENV_SETUP_COMMANDS = ["export PIP_CACHE_DIR=~/.cache/pip"]

# Image validation
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]

# Evaluation configuration
DEFAULT_EVAL_WORKERS = "12"
DEFAULT_MODEL_NAME = "openhands"

# Annotation keywords
SOLVEABLE_KEYWORD = "SOLVEABLE"

# Files to remove from patches during evaluation
SETUP_FILES_TO_REMOVE = ["pyproject.toml", "tox.ini", "setup.py"]

# Annotations file name
ANNOTATIONS_FILENAME = "ambiguity_annotations.json"
