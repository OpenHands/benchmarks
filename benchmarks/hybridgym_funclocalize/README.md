# Hybrid-Gym func_localize Evaluation

This module integrates the **func_localize** benchmark from [Hybrid-Gym](https://github.com/Hybrid-Gym/Hybrid-Gym) ("Hybrid-Gym: Training Coding Agents to Generalize Across Tasks"). The agent must locate a function or class by its natural-language description and write a docstring for it.

## Overview

Each instance provides:
- A Python repository (cloned at runtime from GitHub)
- A description of a target function or class (the original docstring has been removed)

The agent must search the codebase to find the target, understand its implementation, and insert an accurate docstring. Success requires that (1) a docstring was added within the target's line range and (2) all changes are comments or docstrings only (no code modifications).

## Dataset

- **HuggingFace**: [`hybrid-gym/hybrid_gym_func_localize`](https://huggingface.co/datasets/hybrid-gym/hybrid_gym_func_localize)
- **Default split**: `train`
- **Instance types**: Single-function and multi-function (agent must add docstrings to all targets)

Each instance contains:

| Field | Description |
|-------|-------------|
| `instance_id` | Unique identifier (e.g., `internetarchive__openlibrary-462_22`) |
| `repo` | GitHub repository (e.g., `internetarchive/openlibrary`) |
| `base_commit` | Commit SHA to check out |
| `module_name` | Target function/class name |
| `module_type` | `function` or `class` |
| `function_description` | Natural-language description of the target |
| `module_line_start/end` | Ground truth line range (0-indexed) |
| `docstring_line_start/end` | Original docstring line range to remove |
| `functions` | (Multi-function instances) List of targets |

## Prerequisites

1. **Docker**: Required for workspace containers.
2. **LLM API Key**: Configure your LLM provider credentials.

For local (Docker) workspace, the agent server image is built on-the-fly from `python:3.11-bookworm`. For remote workspace, the image must be pre-built and pushed (see below).

## Usage

### LLM Configuration

Create an LLM configuration file (e.g., `.llm_config/config.json`):

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

### Docker Workspace (Local)

```bash
# Run full evaluation
uv run hybridgym-funclocalize-infer .llm_config/config.json --workspace docker

# Limit to 5 instances
uv run hybridgym-funclocalize-infer .llm_config/config.json --workspace docker --n-limit 5
```

### Remote Workspace

The remote workspace requires a pre-built agent server image in a container registry. If the image has already been published to `ghcr.io/openhands/eval-agent-server` (the default), you only need:

```bash
export RUNTIME_API_KEY="<your-openhands-runtime-api-key>"
export IMAGE_TAG_PREFIX="<tag-prefix-matching-the-published-image>"

uv run hybridgym-funclocalize-infer .llm_config/config.json \
    --workspace remote \
    --num-workers 8 \
    --n-limit 5
```

#### Building the image yourself

If the image is not yet available in the default registry, you can build and push to your own:

```bash
docker login ghcr.io -u <GITHUB_USERNAME> --password-stdin <<< "<GITHUB_PAT>"

export OPENHANDS_EVAL_AGENT_SERVER_IMAGE="ghcr.io/<GITHUB_USERNAME>/eval-agent-server"

uv run python -c "
from benchmarks.utils.build_utils import build_image
result = build_image(
    base_image='python:3.11-bookworm',
    target_image='${OPENHANDS_EVAL_AGENT_SERVER_IMAGE}',
    custom_tag='hybridgym-funclocalize',
    target='binary',
    push=True,
)
print('Tags:', result.tags)
"
```

The PAT needs `write:packages` scope. Make the package public at `https://github.com/users/<GITHUB_USERNAME>/packages/container/package/eval-agent-server` (Package settings > Danger Zone > Public).

Then run with:

```bash
export RUNTIME_API_KEY="<your-openhands-runtime-api-key>"
export OPENHANDS_EVAL_AGENT_SERVER_IMAGE="ghcr.io/<GITHUB_USERNAME>/eval-agent-server"
export IMAGE_TAG_PREFIX="$(uv run python -c "from benchmarks.utils.version import SDK_SHORT_SHA; print(SDK_SHORT_SHA)")"

uv run hybridgym-funclocalize-infer .llm_config/config.json \
    --workspace remote \
    --num-workers 8 \
    --n-limit 5
```

### Evaluating Results

```bash
uv run hybridgym-funclocalize-eval ./eval_outputs/.../output.jsonl --run-id my_run
```

This generates `output.report.json` with resolved/unresolved/error instance counts and success rates.

## Evaluation Criteria

An instance is marked **resolved** when both conditions are met:

| Criterion | Description |
|-----------|-------------|
| `target_docstring_edited` | At least one line was added within the target function/class line range |
| `comments_only` | All changes across all files are comments or docstrings (no code modifications, no removed lines) |

## References

- [Hybrid-Gym Paper](https://arxiv.org/abs/2602.16819v1) -- "Hybrid-Gym: Training Coding Agents to Generalize Across Tasks"
- [Hybrid-Gym GitHub](https://github.com/Hybrid-Gym/Hybrid-Gym)
