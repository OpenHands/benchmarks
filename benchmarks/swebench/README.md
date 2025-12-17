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

### Docker Workspace (Local Evaluation)

#### Step 1: Build Docker Images

Before running inference, you need to build Docker images for the SWE-Bench instances. Each instance requires a specific environment setup based on the repository and issue.

```bash
uv run python -m benchmarks.swebench.build_images \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal
```

#### Step 2: Run Inference

Run evaluation using the built Docker images:

```bash
uv run swebench-infer path/to/llm_config.json \
    --dataset princeton-nlp/SWE-bench_Verified \
    --split test \
    --max-iterations 100 \
    --workspace docker
```

**Selecting specific instances:**

You can run evaluation on a specific subset by creating a text file with instance IDs:

```bash
# Create instances.txt with one instance ID per line
echo "django__django-11333" > instances.txt
echo "astropy__astropy-12345" >> instances.txt

# Run with selection
uv run swebench-infer path/to/llm_config.json \
    --select instances.txt \
    --workspace docker
```

### Remote Workspace (Scalable Cloud Evaluation)

Remote workspace enables running evaluations at scale by using a cloud-based runtime API to provision containers. This is ideal for large-scale benchmark runs with high parallelization.

#### Step 1: Pre-build and Push Images

Images must be pre-built and pushed to a **public** container registry before running remote evaluations.

**Option A: Automated Build via PR Label (Recommended)**

1. Create or update a PR in this repository
2. Add one of the following labels to the PR to trigger image builds:
   - `build-swebench-50`: Build 50 images (quick testing, ~5-10 minutes)
   - `build-swebench-200`: Build 200 images (medium testing, ~20-40 minutes)
   - `build-swebench`: Build all images (full evaluation, ~1-2 hours)
3. The GitHub Action will automatically:
   - Build agent-server images for instances in `princeton-nlp/SWE-bench_Verified` (test split)
   - Push images to `ghcr.io/openhands/eval-agent-server` with tags like:
     ```
     ghcr.io/openhands/eval-agent-server:{SDK_SHA}-{INSTANCE_TAG}-source-minimal
     ```
   - The docutils/roman layer is applied in-place (no suffix) for allowlisted repos that need it (currently `sphinx-doc`)
   - Post a comment on [issue #81](https://github.com/OpenHands/benchmarks/issues/81) with the build results

**Option B: Manual Build**

```bash
uv run python -m benchmarks.swebench.build_images \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal \
  --push \
  --max-workers 32
```

**Important Notes:**
- Images must be **publicly accessible** for the remote runtime to pull them
- The SDK SHA is automatically detected from the `vendor/software-agent-sdk` submodule
- Each SWE-Bench instance gets its own unique image tag based on the repository and issue

#### Step 2: Set Up Environment Variables

```bash
# Required: Your runtime API key
export RUNTIME_API_KEY="your-runtime-api-key-here"

# Optional: Override default runtime API URL
export RUNTIME_API_URL="https://runtime.eval.all-hands.dev"

# Optional: Override SDK SHA for image selection
# (defaults to auto-detected from vendor/software-agent-sdk submodule)
export SDK_SHORT_SHA="abc1234"
```

#### Step 3: Run Inference with Remote Workspace

Run evaluation using the remote workspace with high parallelization:

```bash
uv run swebench-infer .llm_config/sonnet-4-5.json \
    --dataset princeton-nlp/SWE-bench_Verified \
    --split test \
    --workspace remote \
    --num-workers 32 \
    --max-iterations 500 \
    --n-limit 200
```

**Command Options Explained:**
- `--workspace remote`: Use remote runtime instead of local Docker
- `--num-workers 32`: Run 32 instances in parallel (adjust based on your quota)
- `--max-iterations 500`: Maximum steps per instance (higher for complex tasks)
- `--n-limit 200`: Limit to first 200 instances (optional, for testing)

**Example: Full-scale Evaluation**

```bash
# Run on all instances with maximum parallelization
uv run swebench-infer .llm_config/sonnet-4-5.json \
    --workspace remote \
    --num-workers 64 \
    --max-iterations 500
```

**Example: Subset Evaluation**

```bash
# Test on a small subset first
echo "django__django-11333" > test_instances.txt
echo "django__django-12155" >> test_instances.txt

uv run swebench-infer .llm_config/sonnet-4-5.json \
    --select test_instances.txt \
    --workspace remote \
    --num-workers 2 \
    --max-iterations 300
```

#### Troubleshooting Remote Workspace

**Error: "RUNTIME_API_KEY environment variable is not set"**
- Solution: Export the `RUNTIME_API_KEY` environment variable before running

**Error: "Agent server image ... does not exist in container registry"**
- Solution: Ensure images are pre-built and pushed using Step 1
- Verify the SDK SHA matches between your local submodule and the built images
- Check that images are publicly accessible in the registry

**Error: "Connection timeout" or API errors**
- Solution: Check your network connectivity
- Verify the `RUNTIME_API_URL` is correct
- Ensure your API key has sufficient quota for the number of workers

### Comparing Docker vs Remote Workspace

| Aspect | Docker Workspace | Remote Workspace |
|--------|-----------------|------------------|
| **Setup** | Simple, no prerequisites | Requires pre-built images + API key |
| **Scale** | Limited by local resources | Hundreds of parallel workers |
| **Speed** | Slower for large evaluations | Much faster with parallelization |
| **Cost** | Local compute only | API usage costs |
| **Use Case** | Development, testing | Production benchmarks, research |

## Evaluation

After running inference (with either workspace type), evaluate the generated patches using the official SWE-Bench evaluation:

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
