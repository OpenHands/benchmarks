"""
SWTBench Constants

This module serves as the single source of truth for all hyperparameters
and constant values used in the SWTBench evaluation workflow.
"""

# =============================================================================
# Docker/Image Related Constants
# =============================================================================

# Docker image prefixes
SWEBENCH_DOCKER_IMAGE_PREFIX = "docker.io/swebench/"
SWTBENCH_DOCKER_IMAGE_PREFIX = "docker.io/swtbench/"

# Agent server image base
AGENT_SERVER_IMAGE_BASE = "ghcr.io/all-hands-ai/agent-server"

# Prebaked evaluation images registry
PREBAKED_REGISTRY = "ghcr.io/openhands/swtbench-eval"

# Build target for agent server images
DEFAULT_BUILD_TARGET = "source-minimal"

# Image tag constants
IMAGE_TAG_LATEST = "latest"
IMAGE_NAME_SEPARATOR = "1776"

# =============================================================================
# Dataset Related Constants
# =============================================================================

# Default dataset for evaluation
DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"

# Default dataset split
DEFAULT_SPLIT = "test"

# Default model name for predictions
DEFAULT_MODEL_NAME = "OpenHands"

# =============================================================================
# Environment Variable Names
# =============================================================================

ENV_SKIP_BUILD = "SKIP_BUILD"
ENV_RUNTIME_API_KEY = "RUNTIME_API_KEY"
ENV_SDK_SHORT_SHA = "SDK_SHORT_SHA"
ENV_RUNTIME_API_URL = "RUNTIME_API_URL"
ENV_REMOTE_RUNTIME_STARTUP_TIMEOUT = "REMOTE_RUNTIME_STARTUP_TIMEOUT"
ENV_SWTBENCH_FORCE_CONDA = "SWTBENCH_FORCE_CONDA"

# =============================================================================
# Default Values
# =============================================================================

# Default value for SKIP_BUILD environment variable
DEFAULT_SKIP_BUILD = "1"

# Default runtime API URL
DEFAULT_RUNTIME_API_URL = "https://runtime.eval.all-hands.dev"

# Default startup timeout in seconds
DEFAULT_STARTUP_TIMEOUT = "600"

# Default number of workers for evaluation
DEFAULT_EVAL_WORKERS = "12"

# Default eval limit for image building
DEFAULT_EVAL_LIMIT = 1

# Default max workers for image building
DEFAULT_BUILD_MAX_WORKERS = 4

# Default max retries for image building
DEFAULT_BUILD_MAX_RETRIES = 2

# Default batch size for image building
DEFAULT_BUILD_BATCH_SIZE = 10

# Default build mode
DEFAULT_BUILD_MODE = "cli"

# Build mode choices
BUILD_MODE_CHOICES = ["api", "cli"]

# =============================================================================
# File/Directory Paths
# =============================================================================

# SWT-bench repository directory name
SWT_BENCH_REPO_DIR = "swt-bench"

# Evaluation results directory name
EVALUATION_RESULTS_DIR = "evaluation_results"

# Report filename
REPORT_FILENAME = "output.report.json"

# Run ID prefix for evaluation
EVAL_RUN_ID_PREFIX = "eval_"

# Eval note prefix
EVAL_NOTE_PREFIX = "SWT-"

# =============================================================================
# Git/Repository Related Constants
# =============================================================================

# SWT-bench repository URL
SWT_BENCH_REPO_URL = "https://github.com/logic-star-ai/swt-bench.git"

# Git user configuration for commits
GIT_USER_EMAIL = "evaluation@openhands.dev"
GIT_USER_NAME = "OpenHands Evaluation"

# =============================================================================
# Patch Processing Constants
# =============================================================================

# Files to remove from patches during post-processing
SETUP_FILES_TO_REMOVE = ["pyproject.toml", "tox.ini", "setup.py"]

# =============================================================================
# Environment Setup Commands
# =============================================================================

# Default environment setup commands
DEFAULT_ENV_SETUP_COMMANDS = ["export PIP_CACHE_DIR=~/.cache/pip"]
