# SWE-fficiency Benchmark Evaluation

This directory contains the implementation for running SWE-fficiency evaluation using OpenHands agents.

## Overview

SWE-fficiency is a benchmark for evaluating repository-level performance optimization on real workloads. The benchmark contains 498 tasks across nine widely used Python repositories (numpy, pandas, scipy, scikit-learn, matplotlib, xarray, sympy, dask, astropy). Given a complete codebase and a slow workload, an agent must investigate code semantics, localize bottlenecks, and produce a patch that improves performance while passing the same unit tests.

## Dataset

- **Source**: [SWE-fficiency](https://swefficiency.com/)
- **HuggingFace**: `swefficiency/swefficiency`
- **Splits**: `test`, `dev`

## Usage

### Docker Workspace (Local Evaluation)

Make sure your Docker daemon is running and you have sufficient disk space (200-500GB recommended).

The evaluation uses pre-built Docker images from `ghcr.io/swefficiency/swefficiency-images`.

```bash
uv run swefficiency-infer path/to/llm_config.json \
    --dataset swefficiency/swefficiency \
    --split test \
    --max-iterations 100 \
    --num-workers 4 \
    --workspace docker
```

#### Resource Limits

For parallel evaluation, CPU and memory limits can be configured:

```bash
uv run swefficiency-infer path/to/llm_config.json \
    --workspace docker \
    --num-workers 4 \
    --num-cpus-per-worker 4 \
    --mem-limit 16g \
    --num-cpus-to-skip 0
```

**Selecting specific instances:**

```bash
# Create instances.txt with one instance ID per line
echo "numpy__numpy-12345" > instances.txt

# Run with selection
uv run swefficiency-infer path/to/llm_config.json \
    --select instances.txt \
    --workspace docker
```

### Remote Workspace (Scalable Cloud Evaluation)

```bash
export RUNTIME_API_KEY="your-runtime-api-key"

uv run swefficiency-infer path/to/llm_config.json \
    --workspace remote \
    --num-workers 32 \
    --max-iterations 500
```

## Evaluation

After running inference, use the official SWE-fficiency benchmark evaluation tools to evaluate the generated patches. See [SWE-fficiency GitHub](https://github.com/swefficiency/swefficiency) for evaluation instructions.

## Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | HuggingFace dataset name | `swefficiency/swefficiency` |
| `--split` | Dataset split | `test` |
| `--workspace` | Workspace type (`docker` or `remote`) | `docker` |
| `--num-workers` | Number of parallel workers | `4` |
| `--max-iterations` | Maximum agent iterations | `500` |
| `--num-cpus-per-worker` | CPUs per Docker container | `4` |
| `--mem-limit` | Memory limit per container | `16g` |
| `--num-cpus-to-skip` | CPUs to reserve at start | `0` |
| `--prompt-path` | Custom prompt template | `prompts/default.j2` |

## References

- [SWE-fficiency Paper](https://arxiv.org/abs/2511.06090)
- [SWE-fficiency Website](https://swefficiency.com/)
