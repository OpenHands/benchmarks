"""
Hybrid-Gym func_gen benchmark configuration.

Task: Agent is given a function signature and docstring (body removed)
and must implement the function body.

Dataset: hybrid-gym/hybrid_gym_func_gen on HuggingFace.
"""

CONDENSER_DEFAULTS = {
    "enable_condenser": False,
    "condenser_max_size": 240,
    "condenser_keep_first": 2,
}

INFER_DEFAULTS = {
    "dataset": "hybrid-gym/hybrid_gym_func_gen",
    "split": "train",
    "max_iterations": 30,
    "tool_preset": "default",
    "num_workers": 16,
    "n_critic_runs": 1,
    **CONDENSER_DEFAULTS,
}
