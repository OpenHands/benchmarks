"""Default configuration constants for TraceTrial-Bench evaluation."""

from typing import Any


TRACETRIAL_REPO_URL = "https://github.com/TraceLite-AI/TraceTrial-Bench.git"
TRACETRIAL_PINNED_COMMIT = "4f94ae64a8e99ef8806200ffdcf496c9d7390f76"
DATASET_NAME = "TraceLite-AI/TraceTrial-Bench"

INFER_DEFAULTS: dict[str, Any] = {
    "dataset": DATASET_NAME,
    "dataset_split": "final",
    "manifest": "Task/final.manifest",
    "max_iterations": 100,
    "num_workers": 4,
    "openhands_version": "1.16.0",
    "use_vendor_sdk": True,
    "resolution_criterion": "agent_pass",
}

EVAL_DEFAULTS: dict[str, Any] = {
    "resolution_criterion": "agent_pass",
}
