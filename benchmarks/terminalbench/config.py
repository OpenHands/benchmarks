"""Terminal-Bench configuration defaults."""

# Default inference settings (only include values actually used by argparse)
INFER_DEFAULTS = {
    "dataset": "terminal-bench-2",
    "output_dir": "./evaluation_outputs",
    "num_workers": 1,
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
