"""
Constants and hyperparameters for Multi-SWE-Bench evaluation.

This module serves as the single source of truth for all constant values
used throughout the Multi-SWE-Bench benchmark implementation.
"""

# =============================================================================
# Dataset Configuration
# =============================================================================

# Default dataset name on HuggingFace
DEFAULT_DATASET = "bytedance-research/Multi-SWE-Bench"

# Default dataset split
DEFAULT_SPLIT = "test"

# Default programming language
DEFAULT_LANGUAGE = "java"

# Default model name for predictions
DEFAULT_MODEL_NAME = "OpenHands"

# =============================================================================
# Docker/Image Configuration
# =============================================================================

# Default Docker image prefix for Multi-SWE-Bench
DEFAULT_DOCKER_IMAGE_PREFIX = "mswebench"

# Default build target for agent server images
DEFAULT_BUILD_TARGET = "source-minimal"

# Environment variable names
DOCKER_IMAGE_PREFIX_ENV_VAR = "EVAL_DOCKER_IMAGE_PREFIX"
LANGUAGE_ENV_VAR = "LANGUAGE"
SKIP_BUILD_ENV_VAR = "MULTI_SWE_BENCH_SKIP_BUILD"

# =============================================================================
# Runtime Configuration
# =============================================================================

# Default runtime API URL for remote workspace
DEFAULT_RUNTIME_API_URL = "https://runtime.eval.all-hands.dev"

# Default startup timeout in seconds
DEFAULT_STARTUP_TIMEOUT = 600

# Environment variable names for runtime configuration
USE_HINT_TEXT_ENV_VAR = "USE_HINT_TEXT"
USE_INSTANCE_IMAGE_ENV_VAR = "USE_INSTANCE_IMAGE"
RUN_WITH_BROWSING_ENV_VAR = "RUN_WITH_BROWSING"
RUNTIME_API_KEY_ENV_VAR = "RUNTIME_API_KEY"
RUNTIME_API_URL_ENV_VAR = "RUNTIME_API_URL"
SDK_SHORT_SHA_ENV_VAR = "SDK_SHORT_SHA"
REMOTE_RUNTIME_STARTUP_TIMEOUT_ENV_VAR = "REMOTE_RUNTIME_STARTUP_TIMEOUT"

# Default values for boolean environment variables
DEFAULT_USE_HINT_TEXT = False
DEFAULT_USE_INSTANCE_IMAGE = True
DEFAULT_RUN_WITH_BROWSING = False

# =============================================================================
# Evaluation Configuration
# =============================================================================

# Default evaluation mode
DEFAULT_EVAL_MODE = "evaluation"

# Default evaluation config values
DEFAULT_FORCE_BUILD = True
DEFAULT_NEED_CLONE = True
DEFAULT_CLEAR_ENV = True
DEFAULT_STOP_ON_ERROR = False
DEFAULT_MAX_WORKERS = 5
DEFAULT_MAX_WORKERS_BUILD_IMAGE = 5
DEFAULT_MAX_WORKERS_RUN_INSTANCE = 5
DEFAULT_LOG_LEVEL = "DEBUG"

# Fix patch run command for evaluation harness
FIX_PATCH_RUN_CMD = (
    'bash -c "apt update ; apt install -y patch ; '
    "sed -i 's@git apply.*@patch --batch --fuzz=5 -p1 -i /home/test.patch;"
    "patch --batch --fuzz=5 -p1 -i /home/fix.patch@g' /home/fix-run.sh ; chmod +x /home/*.sh  ; /home/fix-run.sh\""
)

# =============================================================================
# Workspace Configuration
# =============================================================================

# Default working directory in container
DEFAULT_WORKING_DIR = "/workspace"

# Default environment setup commands
DEFAULT_ENV_SETUP_COMMANDS = ["export PIP_CACHE_DIR=~/.cache/pip"]
