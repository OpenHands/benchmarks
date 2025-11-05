# SWE-Bench Benchmark Evaluation

This directory contains the implementation for running SWE-Bench evaluation using OpenHands agents.

## Overview

SWE-Bench is a benchmark for evaluating AI agents on real-world software engineering tasks derived from GitHub issues. The benchmark tests an agent's ability to understand problem statements, navigate codebases, and generate patches that resolve issues.

## Dataset

- **Source**: Princeton NLP
- **Datasets**: 
  - `princeton-nlp/SWE-bench` - Full dataset
  - `princeton-nlp/SWE-bench_Lite` - Smaller curated subset
  - `princeton-nlp/SWE-bench_Verified` - Verified instances
- **Splits**: `test`, `dev`

## Usage

### Step 1: Build Docker Images

Before running inference, you need to build Docker images for the SWE-Bench instances. Each instance requires a specific environment setup based on the repository and issue.

```bash
uv run python -m benchmarks.swe_bench.build_images \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --image ghcr.io/openhands/agent-server \
  --target source-minimal
```


### Step 2: Run Inference

Run evaluation using the built Docker images:

```bash
uv run python -m benchmarks.swe_bench.run_infer \
    path/to/llm_config.json \
    --dataset princeton-nlp/SWE-bench_Verified \
    --split test \
    --max-iterations 100
```

**Selecting specific instances:**

You can run evaluation on a specific subset by creating a text file with instance IDs:

```bash
# Create instances.txt with one instance ID per line
echo "django__django-11333" > instances.txt
echo "astropy__astropy-12345" >> instances.txt

# Run with selection
uv run python -m benchmarks.swe_bench.run_infer \
    path/to/llm_config.json \
    --select instances.txt
```

### Step 3: Evaluate Results

After running inference, evaluate the generated patches using the official SWE-Bench evaluation:

**Basic evaluation:**

```bash
uv run swebench-eval output.jsonl
```

**Advanced options:**

```bash
# Specify custom dataset and output file
uv run swebench-eval output.jsonl \
  --dataset princeton-nlp/SWE-bench_Lite \
  --output-file results.swebench.jsonl

# Only convert format without running evaluation
uv run swebench-eval output.jsonl --skip-evaluation
```

The evaluation script will:
1. Convert OpenHands output format to SWE-Bench prediction format
2. Run the official SWE-Bench evaluation harness (unless `--skip-evaluation` is used)
3. Report pass/fail results for each instance

## References

- [SWE-Bench Paper](https://arxiv.org/abs/2310.06770)
- [SWE-Bench GitHub](https://github.com/princeton-nlp/SWE-bench)
- [SWE-Bench Leaderboard](https://www.swebench.com/)
