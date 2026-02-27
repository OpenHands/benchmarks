"""
SWE-fficiency hyperparameters and constant values.

This module provides constant values used in the SWE-fficiency evaluation workflow.
For dataset and worker defaults, see config.py (INFER_DEFAULTS).
"""

from typing import Final, Literal


# Docker images
DOCKER_IMAGE_PREFIX: Final[str] = "ghcr.io/swefficiency/swefficiency-images"

# Build target type (matches openhands.agent_server.docker.build.TargetType)
TargetType = Literal["binary", "binary-minimal", "source", "source-minimal"]
BUILD_TARGET_SOURCE_MINIMAL: Final[TargetType] = "source-minimal"
BUILD_TARGET_BINARY: Final[TargetType] = "binary"
DEFAULT_BUILD_TARGET: Final[TargetType] = BUILD_TARGET_SOURCE_MINIMAL

# Git
GIT_USER_EMAIL: Final[str] = "evaluation@openhands.dev"
GIT_USER_NAME: Final[str] = "OpenHands Evaluation"
GIT_COMMIT_MESSAGE: Final[str] = "patch"

# Timeouts
DEFAULT_COMMAND_TIMEOUT: Final[int] = 600
DEFAULT_SANDBOX_TIMEOUT: Final[int] = 3600
