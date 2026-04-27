"""
Hybrid-Gym issue_localize benchmark configuration.

Task: Agent must locate code related to a GitHub issue and add comments
explaining why each location is relevant.

Dataset: SWE-Gym/SWE-Gym-Raw on HuggingFace.
"""

CONDENSER_DEFAULTS = {
    "enable_condenser": False,
    "condenser_max_size": 240,
    "condenser_keep_first": 2,
}

INFER_DEFAULTS = {
    "dataset": "SWE-Gym/SWE-Gym-Raw",
    "split": "train",
    "max_iterations": 30,
    "tool_preset": "default",
    "num_workers": 16,
    "n_critic_runs": 1,
    **CONDENSER_DEFAULTS,
}
