"""
Hybrid-Gym func_localize benchmark configuration.

Task: Agent must locate a function/class by its description (no file path given)
and add a docstring to it.

Dataset: hybrid-gym/hybrid_gym_func_localize on HuggingFace.
"""

# Condenser configuration
CONDENSER_DEFAULTS = {
    "enable_condenser": False,
    "condenser_max_size": 240,
    "condenser_keep_first": 2,
}

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "dataset": "hybrid-gym/hybrid_gym_func_localize",
    "split": "train",
    "max_iterations": 30,
    "tool_preset": "default",
    "num_workers": 16,
    "n_critic_runs": 1,
    **CONDENSER_DEFAULTS,
}
