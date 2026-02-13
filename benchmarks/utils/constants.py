OUTPUT_FILENAME = "output.jsonl"
EVAL_AGENT_SERVER_IMAGE = "ghcr.io/openhands/eval-agent-server"

# Model identifier used in swebench-style prediction entries.
# The swebench harness uses this value to create log directory structures
# (logs/run_evaluation/{run_id}/{model_name_or_path}/{instance_id}/)
# and to name the final evaluation report file ({model_name_or_path}.{run_id}.json).
MODEL_NAME_OR_PATH = "OpenHands"
