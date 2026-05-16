"""
SWE-Smith hyperparameters and constant values.
"""

from typing import Final, Literal


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
