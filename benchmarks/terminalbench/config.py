"""Terminal-Bench configuration defaults."""

# Default inference settings
INFER_DEFAULTS = {
    "dataset": "terminal-bench-2",
    "split": "test",
    "output_dir": "./evaluation_outputs",
    "n_limit": None,  # No limit by default
    "num_workers": 1,
    "max_iterations": 100,
    "max_attempts": 1,
    "max_retries": 3,
    "workspace": "docker",  # docker or remote
}

# Default evaluation settings
EVAL_DEFAULTS = {
    "dataset": "terminal-bench-2",
    "split": "test",
}

# Harbor configuration defaults
HARBOR_DEFAULTS = {
    # Harbor executable
    "harbor_executable": "harbor",
    # Default agent name for openhands-sdk
    "agent_name": "openhands-sdk",
    # Default timeout for agent execution (in seconds)
    "timeout": 3600,  # 1 hour
}

# Terminal-Bench task categories (for reference)
TASK_CATEGORIES = [
    "hello-world",
    "basic",
    "intermediate",
    "advanced",
]
