"""
Hybrid-Gym dep_search benchmark configuration.

Task: Agent must find all functions/classes directly called by a target function
and add a comment above each called module's definition.

Dataset: hybrid-gym/hybrid_gym_dep_search on HuggingFace.
"""

CONDENSER_DEFAULTS = {
    "enable_condenser": False,
    "condenser_max_size": 240,
    "condenser_keep_first": 2,
}

INFER_DEFAULTS = {
    "dataset": "hybrid-gym/hybrid_gym_dep_search",
    "split": "train",
    "max_iterations": 30,
    "tool_preset": "default",
    "num_workers": 16,
    "n_critic_runs": 1,
    **CONDENSER_DEFAULTS,
}
