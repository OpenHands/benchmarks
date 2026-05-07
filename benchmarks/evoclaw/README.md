# EvoClaw

This benchmark entrypoint runs OpenHands against EvoClaw repositories through the
standard OpenHands benchmarks SDK path:

1. discover EvoClaw repo directories from `--data-root`,
2. build/start an OpenHands agent-server workspace from each EvoClaw base image,
3. upload the EvoClaw task queue and SRS files into the workspace,
4. run `Agent`/`Conversation` with the normal fake-user evaluation loop,
5. emit the resulting git patch and conversation trajectory.

```bash
uv run evoclaw-infer .llm_config/example.json \
  --data-root /path/to/EvoClaw-data \
  --repos navidrome \
  --n-limit 1
```

This is currently an inference harness. It intentionally does not reimplement
EvoClaw's milestone DAG grader inside this repo.
