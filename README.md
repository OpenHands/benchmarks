# OpenHands Benchmarks Migration

⚠️ **Migration in Progress**: We are currently migrating the benchmarks infrastructure from [OpenHands](https://github.com/All-Hands-AI/OpenHands/) to work with the [OpenHands Agent SDK](https://github.com/All-Hands-AI/agent-sdk).

## Prerequisites

Before running any benchmarks, run the followings to set up the environment
```bash
make build
```

## Quick Start

### 1. Configure Your LLM

Define your LLM config as a JSON following the model fields type in the [LLM class](https://github.com/All-Hands-AI/agent-sdk/blob/main/openhands/sdk/llm/llm.py#L93), [for example](.llm_config/example.json), you can write the following to `.llm_config/example.json`:

```json
{
  "model": "litellm_proxy/anthropic/claude-sonnet-4-20250514",
  "base_url": "https://llm-proxy.eval.all-hands.dev",
  "api_key": "YOUR_API_KEY_HERE"
}
```

You may validate the correctness of your config by running `uv run validate-cfg .llm_config/YOUR_CONFIG_PATH.json`


### 2. Run SWE-Bench Evaluation

```bash
# Run evaluation with your configured LLM
uv run swebench-infer \
  --llm-config-path .llm_config/default.json \
  --dataset princeton-nlp/SWE-bench-Verified \
  --split test \
  --max-iterations 100 \
  --num-workers 1 \
  --n-limit 10
```

## Links

- **Original OpenHands**: https://github.com/All-Hands-AI/OpenHands/
- **Agent SDK**: https://github.com/All-Hands-AI/agent-sdk
- **SWE-Bench**: https://www.swebench.com/
