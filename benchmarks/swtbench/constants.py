"""
SWTBench Constants

This module serves as the single source of truth for all hyperparameters
and constant values used in the SWTBench evaluation workflow.
"""

from enum import Enum
from typing import Final, Tuple


# =============================================================================
# Docker/Image Related Constants
# =============================================================================

# Docker image prefixes
SWEBENCH_DOCKER_IMAGE_PREFIX: Final[str] = "docker.io/swebench/"
SWTBENCH_DOCKER_IMAGE_PREFIX: Final[str] = "docker.io/swtbench/"

# Agent server image base
AGENT_SERVER_IMAGE_BASE: Final[str] = "ghcr.io/all-hands-ai/agent-server"

# Prebaked evaluation images registry
PREBAKED_REGISTRY: Final[str] = "ghcr.io/openhands/swtbench-eval"

# Build target for agent server images
DEFAULT_BUILD_TARGET: Final[str] = "source-minimal"

# Image tag constants
IMAGE_TAG_LATEST: Final[str] = "latest"
IMAGE_NAME_SEPARATOR: Final[str] = "1776"


class BuildMode(str, Enum):
    """Build mode options for SWT-bench evaluation."""

    API = "api"
    CLI = "cli"


# Default build mode
DEFAULT_BUILD_MODE: Final[str] = BuildMode.CLI.value

# Build mode choices (tuple for immutability)
BUILD_MODE_CHOICES: Final[Tuple[str, ...]] = tuple(m.value for m in BuildMode)

# =============================================================================
# Dataset Related Constants
# =============================================================================

# Default dataset for evaluation
DEFAULT_DATASET: Final[str] = "princeton-nlp/SWE-bench_Verified"

# Default dataset split
DEFAULT_SPLIT: Final[str] = "test"

# Default model name for predictions
DEFAULT_MODEL_NAME: Final[str] = "OpenHands"

# =============================================================================
# Environment Variable Names
# =============================================================================

ENV_SKIP_BUILD: Final[str] = "SKIP_BUILD"
ENV_RUNTIME_API_KEY: Final[str] = "RUNTIME_API_KEY"
ENV_SDK_SHORT_SHA: Final[str] = "SDK_SHORT_SHA"
ENV_RUNTIME_API_URL: Final[str] = "RUNTIME_API_URL"
ENV_REMOTE_RUNTIME_STARTUP_TIMEOUT: Final[str] = "REMOTE_RUNTIME_STARTUP_TIMEOUT"
ENV_SWTBENCH_FORCE_CONDA: Final[str] = "SWTBENCH_FORCE_CONDA"

# =============================================================================
# Default Values
# =============================================================================

# Default value for SKIP_BUILD environment variable (truthy string)
DEFAULT_SKIP_BUILD: Final[str] = "1"

# Default runtime API URL
DEFAULT_RUNTIME_API_URL: Final[str] = "https://runtime.eval.all-hands.dev"

# Default startup timeout in seconds
DEFAULT_STARTUP_TIMEOUT: Final[int] = 600

# Default number of workers for evaluation
DEFAULT_EVAL_WORKERS: Final[int] = 12

# Default eval limit for image building
DEFAULT_EVAL_LIMIT: Final[int] = 1

# Default max workers for image building
DEFAULT_BUILD_MAX_WORKERS: Final[int] = 4

# Default max retries for image building
DEFAULT_BUILD_MAX_RETRIES: Final[int] = 2

# Default batch size for image building
DEFAULT_BUILD_BATCH_SIZE: Final[int] = 10

# =============================================================================
# File/Directory Paths
# =============================================================================

# SWT-bench repository directory name
SWT_BENCH_REPO_DIR: Final[str] = "swt-bench"

# Evaluation results directory name
EVALUATION_RESULTS_DIR: Final[str] = "evaluation_results"

# Report filename
REPORT_FILENAME: Final[str] = "output.report.json"

# Run ID prefix for evaluation
EVAL_RUN_ID_PREFIX: Final[str] = "eval_"

# Eval note prefix
EVAL_NOTE_PREFIX: Final[str] = "SWT-"

# =============================================================================
# Git/Repository Related Constants
# =============================================================================

# SWT-bench repository URL
SWT_BENCH_REPO_URL: Final[str] = "https://github.com/logic-star-ai/swt-bench.git"

# Git user configuration for commits
GIT_USER_EMAIL: Final[str] = "evaluation@openhands.dev"
GIT_USER_NAME: Final[str] = "OpenHands Evaluation"

# =============================================================================
# Patch Processing Constants
# =============================================================================

# Files to remove from patches during post-processing (tuple for immutability)
SETUP_FILES_TO_REMOVE: Final[Tuple[str, ...]] = (
    "pyproject.toml",
    "tox.ini",
    "setup.py",
)

# =============================================================================
# Environment Setup Commands
# =============================================================================

# Default environment setup commands (tuple for immutability)
DEFAULT_ENV_SETUP_COMMANDS: Final[Tuple[str, ...]] = (
    "export PIP_CACHE_DIR=~/.cache/pip",
)
