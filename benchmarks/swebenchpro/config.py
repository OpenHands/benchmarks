"""SWE-Bench Pro benchmark configuration."""

from benchmarks.swebenchpro import constants


INFER_DEFAULTS = {
    "dataset": "ScaleAI/SWE-bench_Pro",
    "split": "test",
    "num_workers": 30,
    "enable_condenser": True,
    "condenser_max_size": 240,
    "condenser_keep_first": 2,
}

EVAL_DEFAULTS = {
    "dataset": "ScaleAI/SWE-bench_Pro",
    "split": "test",
    "workers": 12,
    "dockerhub_username": constants.DEFAULT_DOCKERHUB_USERNAME,
    "use_local_docker": True,
    "block_network": False,
}
