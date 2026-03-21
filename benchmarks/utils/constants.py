import os


OUTPUT_FILENAME = "output.jsonl"

# Image name for agent server (can be overridden via env var)
EVAL_AGENT_SERVER_IMAGE = os.getenv(
    "OPENHANDS_EVAL_AGENT_SERVER_IMAGE", "ghcr.io/openhands/eval-agent-server"
)

# Image name for eval base images (SDK-independent layer).
# These are built once per SWE-bench base image and reused across SDK commits.
EVAL_BASE_IMAGE = os.getenv("OPENHANDS_EVAL_BASE_IMAGE", "ghcr.io/openhands/eval-base")

# Image name for the SDK venv image (built once per SDK commit, shared across all images).
EVAL_SDK_VENV_IMAGE = os.getenv(
    "OPENHANDS_EVAL_SDK_VENV_IMAGE", "ghcr.io/openhands/eval-sdk-venv"
)

# Model identifier used in swebench-style prediction entries.
# The swebench harness uses this value to create log directory structures
# (logs/run_evaluation/{run_id}/{model_name_or_path}/{instance_id}/)
# and to name the final evaluation report file ({model_name_or_path}.{run_id}.json).
MODEL_NAME_OR_PATH = "OpenHands"
