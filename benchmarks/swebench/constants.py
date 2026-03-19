"""
SWE-Bench hyperparameters and constant values.

This module provides constant values used in the SWE-Bench evaluation workflow.
For dataset, model, and worker defaults, see config.py (INFER_DEFAULTS, EVAL_DEFAULTS).
"""

from typing import Final, Literal


# Docker
DOCKER_IMAGE_PREFIX: Final[str] = "docker.io/swebench/"
DOCKER_IMAGE_TAG: Final[str] = "latest"
WRAPPED_REPOS: Final[frozenset[str]] = frozenset(
    {"sphinx-doc"}
)  # Repos requiring docutils/roman wrapper

# Base image tag version -- bump this when the base layer Dockerfile
# (base-image-minimal stage in the SDK Dockerfile) changes so that stale
# cached bases are rebuilt.  The version is prefixed to the eval-base tag:
#   ghcr.io/openhands/eval-base:v1-sweb.eval.x86_64.django_1776_django-12155
BASE_IMAGE_TAG_VERSION: Final[str] = "v1"

# Build target type (matches openhands.agent_server.docker.build.TargetType)
TargetType = Literal["binary", "binary-minimal", "source", "source-minimal"]
BUILD_TARGET_SOURCE_MINIMAL: Final[TargetType] = "source-minimal"
BUILD_TARGET_BINARY: Final[TargetType] = "binary"
DEFAULT_BUILD_TARGET: Final[TargetType] = BUILD_TARGET_SOURCE_MINIMAL

# Runtime
DEFAULT_RUNTIME_API_URL: Final[str] = "https://runtime.eval.all-hands.dev"
DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT: Final[int] = 600


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
