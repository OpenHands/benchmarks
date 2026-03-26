# SWE-Bench Multilingual Benchmark Evaluation

This directory contains the implementation for running SWE-Bench Multilingual evaluation using OpenHands agents.

## Overview

SWE-Bench Multilingual is a benchmark for evaluating AI agents on real-world software engineering tasks from non-English GitHub repositories. The benchmark tests an agent's ability to understand problem statements in multiple languages, navigate codebases with non-English comments and documentation, and generate patches that resolve issues.

## Dataset

- **Source**: SWE-bench organization
- **Dataset**: `SWE-bench/SWE-bench_Multilingual`
- **Splits**: `test`

## Usage

### Docker Workspace (Local Evaluation)

#### Step 1: Build Docker Images

Before running inference, you need to build Docker images for the SWE-Bench Multilingual instances. Each instance requires a specific environment setup based on the repository and issue.

```bash
uv run python -m benchmarks.swebenchmultilingual.build_images \
  --dataset SWE-bench/SWE-bench_Multilingual \
  --split test \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal
```

#### Step 2: Run Inference

Run evaluation using the built Docker images:

```bash
uv run swebenchmultilingual-infer path/to/llm_config.json \
    --dataset SWE-bench/SWE-bench_Multilingual \
    --split test \
    --max-iterations 100 \
    --workspace docker
```

You can resume a previous run by re-running the same command with the same `--output-dir`. Previously completed instances are automatically skipped.

**Selecting specific instances:**

You can run evaluation on a specific subset by creating a text file with instance IDs:

```bash
# Create instances.txt with one instance ID per line
echo "django__django-11333" > instances.txt

# Run with selection
uv run swebenchmultilingual-infer path/to/llm_config.json \
    --select instances.txt \
    --workspace docker
```

### Remote Workspace (Scalable Cloud Evaluation)

Remote workspace enables running evaluations at scale by using a cloud-based runtime API to provision containers.

#### Step 1: Run Inference with Remote Workspace

```bash
uv run swebenchmultilingual-infer path/to/llm_config.json \
    --dataset SWE-bench/SWE-bench_Multilingual \
    --split test \
    --max-iterations 100 \
    --workspace remote
```

### Evaluation

After running inference, evaluate the results:

```bash
uv run swebenchmultilingual-eval <path_to_output.jsonl>
```

This will:
1. Convert the OpenHands output format to SWE-Bench prediction format
2. Run the SWE-Bench evaluation with the appropriate settings
3. Generate a cost report

Example:
```bash
uv run swebenchmultilingual-eval ./output/output.jsonl --workers 8
```

For more evaluation options:
```bash
uv run swebenchmultilingual-eval --help
```

## Configuration

The benchmark uses similar configuration options as regular SWE-Bench:

- `--dataset`: Dataset name (should be `SWE-bench/SWE-bench_Multilingual`)
- `--split`: Dataset split (e.g., `test`)
- `--llm-config`: Path to LLM configuration file
- `--max-iterations`: Maximum number of agent iterations
- `--workspace`: Either `docker` or `remote`
- `--num-workers`: Number of parallel workers

## Environment Variables

- `SKIP_BUILD=1`: Skip building docker images (use pre-built images)
- `RUNTIME_API_KEY`: Required for remote workspace
- `RUNTIME_API_URL`: Runtime API URL (defaults to https://runtime.eval.all-hands.dev)

## Multilingual Considerations

When working with multilingual instances:

- Problem statements may be in various languages
- Code comments and documentation may be multilingual
- Test output and error messages may be in non-English languages
- The agent should be able to handle multilingual contexts effectively
