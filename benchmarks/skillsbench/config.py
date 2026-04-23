"""SkillsBench configuration defaults."""

# Default inference settings
INFER_DEFAULTS = {
    "dataset": "benchflow/skillsbench",
    "output_dir": "./evaluation_outputs",
    "num_workers": 1,
}

# benchflow configuration defaults
BENCHFLOW_DEFAULTS = {
    "agent_name": "openhands",
}
