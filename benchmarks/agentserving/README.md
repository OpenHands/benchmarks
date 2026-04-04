# Agent Serving Benchmark

This benchmark profiles how well an LLM endpoint serves **multiple concurrent OpenHands SDK agents** running the same local-runtime coding task.

Each agent starts in a fresh random directory under `/tmp`, receives the task below, and runs with the standard OpenHands local tool stack (terminal, file editor, task tracker):

> In directory X, create a simple and modular library for generating an HTML page from Python documentation, and apply the library to itself.

The benchmark is designed for **serving experiments** rather than accuracy leaderboards. It measures:

- end-to-end batch wall clock time
- per-agent completion/failure rate
- per-agent LLM call latencies from the OpenHands SDK
- prompt/completion token throughput
- optional server-side vLLM metrics like TTFT and e2e latency via `/metrics`

## Run the benchmark against any OpenAI-compatible endpoint

1. Create an LLM config JSON:

```json
{
  "model": "openai/nemotron-3-super",
  "base_url": "https://your-server.example.com/v1",
  "api_key": "EMPTY"
}
```

For vLLM and other OpenAI-compatible servers, use the served model name with an
`openai/` prefix so LiteLLM routes the request through its OpenAI-compatible
provider.

2. Run a sweep over parallelism levels:

```bash
uv run agentserving-infer .llm_config/your-model.json \
  --parallelism-levels 1 2 4 8 16 32 \
  --machine-size 4xh100 \
  --context-length-k 32 \
  --note nemotron_4xh100_32k
```

The runner derives `/health` and `/metrics` from the LLM `base_url` automatically when it ends in `/v1`.

## Output layout

Results are written under a structured directory in `eval_outputs/agentserving/...`.

Each parallelism level gets its own folder with:

- `experiment.json`: full structured record
- `results.jsonl`: one JSON record per agent
- `conversations/`: persisted OpenHands conversation state per agent

The root run directory also contains:

- `sweep.json`: aggregate summaries for all tested parallelism levels
- `sweep.jsonl`: one summary record per parallelism level

## Modal + vLLM helper

This benchmark includes a Modal helper for self-hosting a vLLM server on H100s.

### Supported presets

- `nemotron-3-super` → `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8`
- `minimax-m2.5` → `MiniMaxAI/MiniMax-M2.5`

### Deploy to Modal

The helper is configured through environment variables so the same file can be reused for different models, machine sizes, and context lengths.

```bash
export MODAL_CONFIG_PATH=/workspace/project/modal.toml
export AGENTSERVING_MODEL_ALIAS=nemotron-3-super
export AGENTSERVING_MACHINE_SIZE=4xh100
export AGENTSERVING_CONTEXT_LENGTH_K=32
export AGENTSERVING_APP_NAME=agentserving-nemotron-4xh100-32k

HF_TOKEN=$HF_TOKEN \
OPENHANDS_SUPPRESS_BANNER=1 \
uv run modal deploy benchmarks/agentserving/modal_vllm.py
```

For MiniMax M2.5 on 8xH100, switch:

```bash
export AGENTSERVING_MODEL_ALIAS=minimax-m2.5
export AGENTSERVING_MACHINE_SIZE=8xh100
export AGENTSERVING_CONTEXT_LENGTH_K=128
export AGENTSERVING_APP_NAME=agentserving-minimax-m25-8xh100-128k
```

### Benchmark the deployed server

Create an LLM config that points at the deployed Modal URL and uses the served
model name with an `openai/` prefix, then run the sweep:

```bash
uv run agentserving-infer .llm_config/modal-nemotron.json \
  --parallelism-levels 1 2 4 8 16 32 \
  --machine-size 4xh100 \
  --context-length-k 32 \
  --note modal_nemotron_4xh100_32k
```

## Notes on the serving recipes

### Nemotron 3 Super FP8

This helper follows the public vLLM/NVIDIA guidance for H100 serving:

- FP8 weights
- FP8 KV cache
- tensor parallelism sized to the node (`4` or `8`)
- `qwen3_coder` tool parser
- `nemotron_v3` reasoning parser

### MiniMax M2.5

This helper follows the public vLLM MiniMax M2-series guidance:

- 4xH100: tensor parallel 4
- 8xH100: tensor parallel 8 + expert parallel
- `minimax_m2` tool parser
- `minimax_m2_append_think` reasoning parser

If you encounter the H100 cudagraph issue mentioned in the vLLM docs, redeploy with:

```bash
export AGENTSERVING_PIECEWISE_CUDAGRAPH=1
```
