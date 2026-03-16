# Terminal-Bench Evaluation

This module provides integration with [Terminal-Bench](https://tbench.ai), a benchmark for evaluating AI agents on terminal-based tasks. The integration uses [Harbor](https://harborframework.com) as the evaluation harness with the `openhands-sdk` agent.

## Overview

Terminal-Bench evaluates how well AI agents can handle real-world, end-to-end tasks in command-line environments, including:
- Compiling code
- Training models
- Setting up servers
- System administration tasks

## Prerequisites

1. **Install Harbor**: Harbor is the official harness for running Terminal-Bench 2.0.

```bash
pip install harbor-bench
# or
uv pip install harbor-bench
```

2. **Docker**: Harbor requires Docker to be installed and running.

3. **LLM API Key**: Configure your LLM provider credentials.

## Usage

### Running Inference

Run the Terminal-Bench evaluation using the OpenHands SDK agent:

```bash
# Run full evaluation
uv run terminalbench-infer .llm_config/claude.json

# Run specific tasks
uv run terminalbench-infer .llm_config/claude.json --task-id hello-world

# Run tasks from a file
uv run terminalbench-infer .llm_config/claude.json --select tasks.txt

# Run with specific dataset version
uv run terminalbench-infer .llm_config/claude.json --dataset terminal-bench@2.0

# Run with multiple workers
uv run terminalbench-infer .llm_config/claude.json --num-workers 4
```

### LLM Configuration

Create an LLM configuration file (e.g., `.llm_config/claude.json`):

```json
{
  "model": "anthropic/claude-sonnet-4-20250514",
  "api_key": "YOUR_API_KEY"
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
uv run terminalbench-eval ./evaluation_outputs/.../output.jsonl
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
  "instance_id": "task-name",
  "test_result": {
    "trajectory_path": "path/to/trajectory.json",
    "total_steps": 15,
    "final_metrics": {
      "total_prompt_tokens": 5000,
      "total_completion_tokens": 1000,
      "total_cost_usd": 0.05
    }
  },
  "instruction": "Task description...",
  "history": [...],
  "metrics": {...}
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

The integration follows the Harbor agent adapter pattern:

1. **Harbor Harness**: Manages task containers and lifecycle
2. **OpenHands SDK Agent**: Runs inside containers to solve tasks
3. **ATIF Trajectories**: Results stored in Agent Trajectory Interchange Format

```
┌──────────────────────────────────────────────────┐
│                 Harbor Harness                   │
│  ┌────────────────────────────────────────────┐  │
│  │           Task Container                   │  │
│  │  ┌──────────────────────────────────────┐  │  │
│  │  │       OpenHands SDK Agent            │  │  │
│  │  │  - Terminal tool                     │  │  │
│  │  │  - File editor tool                  │  │  │
│  │  │  - Task tracker tool                 │  │  │
│  │  └──────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

## References

- [Terminal-Bench](https://tbench.ai) - The benchmark
- [Harbor](https://harborframework.com) - The evaluation harness
- [OpenHands SDK](https://github.com/OpenHands/software-agent-sdk) - The agent SDK
- [ATIF Specification](https://github.com/laude-institute/harbor/blob/main/docs/rfcs/0001-trajectory-format.md) - Trajectory format
