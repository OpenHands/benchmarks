"""SkillsBench configuration defaults."""

from benchmarks.utils.harbor_compat import get_harbor_dataset


# Default inference settings (only include values actually used by argparse)
INFER_DEFAULTS = {
    "dataset": get_harbor_dataset("skillsbench"),
    "output_dir": "./evaluation_outputs",
    "num_workers": 1,
}

# Harbor configuration defaults
HARBOR_DEFAULTS = {
    # Harbor executable
    "harbor_executable": "harbor",
    # Default agent name for openhands-sdk
    "agent_name": "openhands-sdk",
}
