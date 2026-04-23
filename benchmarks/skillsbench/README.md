# SkillsBench Evaluation

This module provides integration with [SkillsBench](https://www.skillsbench.ai/), a benchmark for evaluating AI agents on real-world skill-based tasks. The integration uses [benchflow](https://github.com/benchflow-ai/benchflow) as the evaluation harness with the `openhands` agent.

## Overview

SkillsBench comprises tasks across 11 domains, evaluating the efficacy of Skills augmentation in LLM-based agents. Domains include:

- Software engineering
- Office & white collar
- Natural science
- Media & content production
- Cybersecurity
- Finance
- Robotics
- Manufacturing
- Energy
- Mathematics
- Healthcare

## Prerequisites

1. **Install benchflow**: benchflow is the official harness for running SkillsBench.

   ```bash
   uv tool install benchflow==0.3.0
   # or
   pip install benchflow==0.3.0
   # or
   uv pip install benchflow==0.3.0
   ```

2. **Docker**: benchflow requires Docker to be installed and running.

3. **LLM API Key**: Configure your LLM provider credentials. The benchflow `openhands` agent reads `LLM_API_KEY` and optional `LLM_BASE_URL` from the environment.

## Usage

### Running Inference

Run the SkillsBench evaluation using the `openhands` agent:

```bash
# Run full evaluation
uv run skillsbench-infer .llm_config/claude.json

# Run specific tasks
uv run skillsbench-infer .llm_config/claude.json --task-id benchflow/weighted-gdp-calc

# Run tasks from a file
uv run skillsbench-infer .llm_config/claude.json --select tasks.txt

# Limit the run to 5 tasks (useful for smoke tests)
uv run skillsbench-infer .llm_config/claude.json --n-limit 5

# Run with multiple workers
uv run skillsbench-infer .llm_config/claude.json --num-workers 4
```

### LLM Configuration

Create an LLM configuration file (e.g., `.llm_config/claude.json`):

```json
{
  "model": "anthropic/claude-sonnet-4-20250514",
  "api_key": "YOUR_ANTHROPIC_API_KEY"
}
```

Or use a LiteLLM proxy:

```json
{
  "model": "litellm_proxy/anthropic/claude-sonnet-4-20250514",
  "base_url": "https://your-proxy.example.com",
  "api_key": "YOUR_API_KEY"
}
```

### Evaluating Results

After running inference, evaluate the results:

```bash
uv run skillsbench-eval ./evaluation_outputs/.../output.jsonl
```

This generates a report file (`output.report.json`) with:
- Total/completed/resolved instance counts
- Success rate
- Aggregate metrics (cost, tokens)

## Output Format

### Inference Output (`output.jsonl`)

Each line contains:

```json
{
  "instance_id": "benchflow/task-name",
  "test_result": {
    "rewards": {"reward": 1.0},
    "passed": true
  },
  "instruction": "",
  "error": null,
  "history": [],
  "metrics": {
    "total_prompt_tokens": 5000,
    "total_completion_tokens": 1000,
    "total_cost_usd": 0.05
  }
}
```

### Evaluation Report (`output.report.json`)

```json
{
  "total_instances": 100,
  "completed_instances": 95,
  "resolved_instances": 80,
  "unresolved_instances": 15,
  "error_instances": 5,
  "aggregate_metrics": {
    "total_cost_usd": 5.25,
    "total_prompt_tokens": 500000,
    "total_completion_tokens": 100000
  }
}
```

## Architecture

The integration uses the benchflow CLI as the evaluation harness:

1. **Task download**: the integration clones the SkillsBench task repo locally when the task cache is empty
2. **benchflow job**: Runs all tasks concurrently with `openhands`
3. **Result conversion**: Trial `result.json` files are converted to the standard `output.jsonl` format

```text
┌──────────────────────────────────────────────────┐
│               benchflow job                      │
│  ┌────────────────────────────────────────────┐  │
│  │           Task Container (Docker)          │  │
│  │  ┌──────────────────────────────────────┐  │  │
│  │  │       openhands                      │  │  │
│  │  │  - Terminal tool                     │  │  │
│  │  │  - File editor tool                  │  │  │
│  │  └──────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

## References

- [SkillsBench](https://www.skillsbench.ai/) - The benchmark
- [benchflow](https://github.com/benchflow-ai/benchflow) - The evaluation harness
- [benchflow CLI reference](https://github.com/benchflow-ai/benchflow/blob/main/docs/cli-reference.md) - CLI documentation
