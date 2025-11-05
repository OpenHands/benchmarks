# SWE-Bench Benchmark Evaluation

This directory contains the implementation for running SWE-Bench (Software Engineering Benchmark) evaluation using OpenHands agents.

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

**Build all images for a dataset:**

```bash
uv run python -m benchmarks.swe_bench.build_images \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --critic pass \
  --image ghcr.io/openhands/agent-server \
  --target binary-minimal
```

**Available targets:**
- `binary-minimal` - Minimal binary distribution (recommended, faster)
- `source-minimal` - Build from source

**Options:**
- `--dataset`: Dataset to use (default: `princeton-nlp/SWE-bench_Lite`)
- `--split`: Dataset split (default: `test`)
- `--critic`: Critic to use (default: `pass`)
- `--image`: Base agent server image (default: `ghcr.io/openhands/agent-server`)
- `--target`: Build target (default: `binary-minimal`)

### Step 2: Run Inference

Run evaluation using the built Docker images:

**Basic inference:**

```bash
uv run swebench-infer path/to/llm_config.json
```

**Advanced options:**

```bash
uv run python -m benchmarks.swe_bench.run_infer \
    path/to/llm_config.json \
    --dataset princeton-nlp/SWE-bench_Lite \
    --split test \
    --max-iterations 30 \
    --critic pass \
    --docker-image-prefix docker.io/swebench/ \
    --output-dir outputs/swebench
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

## Configuration Options

### Required Arguments

- LLM config path: Path to JSON configuration file for the language model

### Optional Arguments

- `--dataset`: SWE-Bench dataset to use (default: `princeton-nlp/SWE-bench_Lite`)
- `--split`: Dataset split (default: `test`)
- `--max-iterations`: Maximum iterations per instance (default: 30)
- `--output-dir`: Base directory for outputs (default: `outputs`)
- `--critic`: Critic to use for evaluation (default: `pass`)
- `--docker-image-prefix`: Docker image prefix (default: `docker.io/swebench/`)
- `--target`: Build target for agent server (default: `binary-minimal`)
- `--select`: File containing instance IDs to evaluate (one per line)
- `--n-limit`: Limit number of instances (default: 0 = all)
- `--num-workers`: Number of parallel workers (default: 1)

## Output Format

Results are written to JSONL files in the output directory. Each line contains:

```json
{
  "instance_id": "repo__name-12345",
  "test_result": {
    "resolved": true,
    "patch": "diff --git a/file.py...",
    "test_output": "..."
  },
  "instruction": "The issue description...",
  "history": [...],
  "instance": {...}
}
```

For evaluation, the output is converted to SWE-Bench format:

```json
{
  "instance_id": "repo__name-12345",
  "model_patch": "diff --git a/file.py...",
  "model_name_or_path": "..."
}
```

## Docker Image Structure

The SWE-Bench evaluation uses a two-layer Docker image approach:

1. **Base SWE-Bench Image**: Contains the repository at the correct commit with test environment
2. **Agent Server Image**: Adds OpenHands agent server on top of the base image

This ensures each instance runs in an isolated, reproducible environment that matches the original repository state.

## References

- [SWE-Bench Paper](https://arxiv.org/abs/2310.06770)
- [SWE-Bench GitHub](https://github.com/princeton-nlp/SWE-bench)
- [SWE-Bench Website](https://www.swebench.com/)
- [SWE-Bench Leaderboard](https://www.swebench.com/)
