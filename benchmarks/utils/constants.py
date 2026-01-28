"""
Shared constants for all benchmarks.

This module contains default values used across multiple benchmarks.
Benchmark-specific constants should be defined in their own constants.py files.
"""

from typing import Final


# Output
OUTPUT_FILENAME: Final[str] = "output.jsonl"

# Docker
EVAL_AGENT_SERVER_IMAGE: Final[str] = "ghcr.io/openhands/eval-agent-server"

# Workspace
DEFAULT_WORKSPACE: Final[str] = "remote"
DEFAULT_SPLIT: Final[str] = "test"

# Evaluation
DEFAULT_MAX_ITERATIONS: Final[int] = 100
DEFAULT_NUM_EVAL_WORKERS: Final[int] = 1
DEFAULT_OUTPUT_DIR: Final[str] = "./eval_outputs"
DEFAULT_MAX_ATTEMPTS: Final[int] = 3
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_NOTE: Final[str] = "initial"
DEFAULT_N_LIMIT: Final[int] = 0

# Critic
DEFAULT_CRITIC: Final[str] = "pass"
