"""
Shared constants for all benchmarks.

This file contains constant values that are shared across multiple benchmarks.
Benchmark-specific constants should be defined in {benchmark}/constants.py.
"""

# Output file configuration
OUTPUT_FILENAME = "output.jsonl"

# Docker image configuration
EVAL_AGENT_SERVER_IMAGE = "ghcr.io/openhands/eval-agent-server"

# Default dataset configuration
DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"

# Default evaluation parameters
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_RETRIES = 3
DEFAULT_NUM_WORKERS = 1
DEFAULT_EVAL_LIMIT = 0

# Workspace configuration
DEFAULT_WORKSPACE_TYPE = "docker"
DEFAULT_OUTPUT_DIR = "./eval_outputs"

# Remote runtime configuration
DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT = 600
DEFAULT_RUNTIME_API_URL = "https://runtime.eval.all-hands.dev"

# Environment setup commands (shared across benchmarks)
DEFAULT_ENV_SETUP_COMMANDS = ["export PIP_CACHE_DIR=~/.cache/pip"]

# Critic configuration
DEFAULT_CRITIC = "pass"
