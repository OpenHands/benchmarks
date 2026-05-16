# SWE-Smith Benchmark Evaluation

This directory contains the implementation for running SWE-Smith evaluation using OpenHands agents.

## Overview

SWE-Smith is a benchmark for training and evaluating AI agents on synthetically generated software engineering tasks. Task instances are created by injecting bugs into real repositories and validating them against test suites.

## Dataset

- **Source**: [SWE-Smith Paper](https://arxiv.org/abs/2504.21798)
- **Dataset**: `SWE-bench/SWE-smith-py`
- **Splits**: `train`
- Local task instance files (`.json` / `.jsonl`) generated via SWE-Smith are also supported.

## Usage

### Step 1: Build Docker Images

Before running inference, you need to build Docker images for the SWE-Smith instances. Each instance requires a specific environment setup. Disk usage depends on the number and size of task instances — the full dataset can consume 150-200GB, but smaller local instance files will use proportionally less.

```bash
uv run python -m benchmarks.swesmith.build_images \
  --dataset SWE-bench/SWE-smith-py \
  --split train \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal
```

For local task instance files:

```bash
uv run python -m benchmarks.swesmith.build_images \
  --dataset /path/to/task_instances.json \
  --split train \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal \
  --n-limit 10
```

### Step 2: Run Inference

```bash
uv run swesmith-infer path/to/llm_config.json \
  --dataset /path/to/task_instances.json \
  --workspace docker \
  --max-iterations 75 \
  --num-workers 4
```

**Selecting specific instances:**

```bash
# Create instances.txt with one instance ID per line
echo "encode__httpx.ae1b9f66.lm_modify__abc123" > instances.txt

uv run swesmith-infer path/to/llm_config.json \
  --dataset /path/to/task_instances.json \
  --select instances.txt \
  --workspace docker
```

### Configuration Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | HuggingFace dataset name or local file path | `SWE-bench/SWE-smith-py` |
| `--split` | Dataset split | `train` |
| `--workspace` | Workspace type | `docker` |
| `--num-workers` | Parallel workers | `4` |
| `--max-iterations` | Max agent turns per instance | `500` |
| `--n-limit` | Limit number of instances | all |
| `--select` | Text file with instance IDs (one per line) | - |
| `--max-attempts` | Retry attempts with critic | `3` |
| `--critic` | `pass` / `finish_with_patch` / `empty_patch_critic` | `finish_with_patch` |
| `--prompt-path` | Jinja2 prompt template | `prompts/default.j2` |
| `--note` | Note appended to output directory name | - |

### Private Repositories

For private repos, an SSH key must be accessible. The lookup order is:

1. `GITHUB_USER_SSH_KEY` environment variable (path to key file)
2. `~/.ssh/id_rsa`, `id_ecdsa`, `id_ecdsa_sk`, `id_ed25519`, `id_ed25519_sk` (first match)

```bash
# Only needed if your key has a non-standard name
export GITHUB_USER_SSH_KEY=~/.ssh/my_custom_key
```

### Environment Variables

Environment variables can be set directly or via a `.env` file in the project root.

All environment variables prefixed with `OPENHANDS_` are forwarded into the Docker container with the prefix stripped. For example, `OPENHANDS_ANTHROPIC_API_KEY` becomes `ANTHROPIC_API_KEY` inside the container. This is how you pass LLM API keys and other credentials to the agent.

```bash
export OPENHANDS_ANTHROPIC_API_KEY=sk-xxx
export OPENHANDS_OPENAI_API_KEY=sk-xxx
export OPENHANDS_GOOGLE_APPLICATION_CREDENTIALS='{"type":"service_account",...}'
```

| Variable | Description |
|----------|-------------|
| `OPENHANDS_*` | Forwarded into the container with prefix stripped (LLM keys, credentials, etc.) |
| `GITHUB_USER_SSH_KEY` | Path to SSH key for private repos |
| `SKIP_BUILD` | Set to `1` to skip Docker image building during inference (default: `1`) |

## Evaluation

After running inference, evaluate the generated patches:

```bash
uv run swesmith-eval output.jsonl \
  --run-id my_eval \
  --dataset /path/to/task_instances.json
```

**Advanced options:**

```bash
# Faster evaluation using only fail-to-pass tests
uv run swesmith-eval output.jsonl \
  --run-id my_eval \
  --dataset /path/to/task_instances.json \
  --f2p-only

# Re-evaluate failed/errored instances
uv run swesmith-eval output.jsonl \
  --run-id my_eval \
  --dataset /path/to/task_instances.json \
  --redo-existing

# Only regenerate the report from existing evaluation logs
uv run swesmith-eval output.jsonl \
  --run-id my_eval \
  --dataset /path/to/task_instances.json \
  --report-only
```

## Output Structure

```
eval_outputs/
└── <dataset>-<split>/
    └── <model>/
        ├── output.jsonl                    # Main results
        ├── output.critic_attempt_N.jsonl   # Per-attempt results
        ├── output.swesmith.jsonl           # SWE-Smith format predictions
        ├── output.report.json              # Evaluation report (SWE-Smith format)
        ├── cost_report.jsonl               # Token usage and cost
        └── conversations/                  # Per-instance conversation logs
            └── <instance_id>.tar.gz
```

**Inference result** (`output.jsonl`, one entry per line):

```json
{
  "instance_id": "encode__httpx.ae1b9f66.lm_modify__abc123",
  "attempt": 1,
  "test_result": {
    "git_patch": "diff --git a/file.py b/file.py\n..."
  },
  "instruction": "...",
  "history": [],
  "metrics": {},
  "error": null
}
```

**Evaluation report** (`output.report.json`) follows the SWE-Smith report format:

```json
{
  "resolved": 5,
  "unresolved": 3,
  "total": 8,
  "ids_resolved": ["instance_1", "..."],
  "ids_unresolved": ["instance_3", "..."]
}
```

## Custom Repository Profiles

To add a custom repository, define a profile class in `profiles.py`:

```python
@dataclass
class MyRepoBcd12345(PythonProfile):
    owner: str = "github-org"
    repo: str = "my-repo"
    commit: str = "bcd1234567890"
    org_gh: str = "org-swesmith"
```

Profiles are auto-registered on import. For Go repositories, inherit from `GoProfile` instead.

## References

- [SWE-Smith Paper](https://arxiv.org/abs/2504.21798)
- [SWE-Smith GitHub](https://github.com/SWE-bench/SWE-smith)
- [SWE-Smith Dataset on HuggingFace](https://huggingface.co/datasets/SWE-bench/SWE-smith)
