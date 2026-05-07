"""Defaults for EvoClaw inference."""

INFER_DEFAULTS = {
    "dataset": "evoclaw",
    "split": "test",
    "max_iterations": 3000,
    "num_workers": 1,
    "n_critic_runs": 1,
    "workspace": "docker",
    "enable_condenser": True,
    "condenser_max_size": 100,
    "condenser_keep_first": 4,
}
