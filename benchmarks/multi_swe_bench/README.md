# Multi-SWE-Bench Evaluation

Multi-SWE-Bench is a multi-language extension of the SWE-Bench benchmark that evaluates software engineering capabilities across multiple programming languages including Java, Python, Go, and C.

## Overview

This benchmark evaluates an agent's ability to:
- Understand and fix software bugs across different programming languages
- Navigate and modify codebases in various languages
- Run language-specific tests and validation
- Generate appropriate patches for different language ecosystems

## Dataset

- **Source**: Bytedance Research
- **Datasets**: 
  - `bytedance-research/Multi-SWE-Bench` - Full multi-language dataset
- **Splits**: `java_verified`, `python_verified`, `go_verified`, `c_verified`, `test`, `dev`

## Supported Languages

- **Java**: Enterprise applications, libraries, and frameworks
- **Python**: Scientific computing, web frameworks, and utilities
- **Go**: System tools, web services, and cloud applications
- **C**: System programming, embedded software, and performance-critical applications

## Usage

### Docker Workspace (Local Evaluation)

#### Step 1: Build Docker Images

Before running inference, you need to build Docker images for the Multi-SWE-Bench instances. Each instance requires a specific environment setup based on the repository and language.

```bash
# Build images for Java instances
LANGUAGE=java uv run python -m benchmarks.multi_swe_bench.build_images \
  --dataset bytedance-research/Multi-SWE-Bench \
  --split java_verified \
  --image ghcr.io/openhands/agent-server \
  --target source-minimal

# Build images for Python instances
LANGUAGE=python uv run python -m benchmarks.multi_swe_bench.build_images \
  --dataset bytedance-research/Multi-SWE-Bench \
  --split python_verified \
  --image ghcr.io/openhands/agent-server \
  --target source-minimal
```

#### Step 2: Run Inference

Run evaluation using the built Docker images:

```bash
# Run inference for Java projects
LANGUAGE=java uv run multi-swebench-infer path/to/llm_config.json \
    --dataset bytedance-research/Multi-SWE-Bench \
    --split java_verified \
    --max-iterations 100 \
    --workspace docker

# Run inference for Python projects
LANGUAGE=python uv run multi-swebench-infer path/to/llm_config.json \
    --dataset bytedance-research/Multi-SWE-Bench \
    --split python_verified \
    --max-iterations 100 \
    --workspace docker
```

**Selecting specific instances:**

You can run evaluation on a specific subset by creating a text file with instance IDs:

```bash
# Create instances.txt with one instance ID per line
echo "apache__commons-cli__CLI-291" > instances.txt
echo "google__gson__Gson-1043" >> instances.txt

# Run with selection
LANGUAGE=java uv run multi-swebench-infer path/to/llm_config.json \
    --select instances.txt \
    --workspace docker
```

### Remote Workspace (Scalable Cloud Evaluation)

Remote workspace enables running evaluations at scale by using a cloud-based runtime API to provision containers. This is ideal for large-scale benchmark runs with high parallelization.

#### Step 1: Pre-build and Push Images

Images must be pre-built and pushed to a **public** container registry before running remote evaluations.

```bash
# Build and push Java images
LANGUAGE=java uv run python -m benchmarks.multi_swe_bench.build_images \
  --dataset bytedance-research/Multi-SWE-Bench \
  --split java_verified \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal \
  --push \
  --max-workers 32

# Build and push Python images
LANGUAGE=python uv run python -m benchmarks.multi_swe_bench.build_images \
  --dataset bytedance-research/Multi-SWE-Bench \
  --split python_verified \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal \
  --push \
  --max-workers 32
```

#### Step 2: Set Up Environment Variables

```bash
# Required: Your runtime API key
export RUNTIME_API_KEY="your-runtime-api-key-here"

# Required: Target programming language
export LANGUAGE="java"  # or python, go, c

# Optional: Override default runtime API URL
export RUNTIME_API_URL="https://runtime.eval.all-hands.dev"

# Optional: Override SDK SHA for image selection
export SDK_SHORT_SHA="abc1234"
```

#### Step 3: Run Inference with Remote Workspace

Run evaluation using the remote workspace with high parallelization:

```bash
# Java evaluation with remote workspace
LANGUAGE=java uv run multi-swebench-infer .llm_config/sonnet-4-5.json \
    --dataset bytedance-research/Multi-SWE-Bench \
    --split java_verified \
    --workspace remote \
    --num-workers 32 \
    --max-iterations 500 \
    --n-limit 200

# Python evaluation with remote workspace
LANGUAGE=python uv run multi-swebench-infer .llm_config/sonnet-4-5.json \
    --dataset bytedance-research/Multi-SWE-Bench \
    --split python_verified \
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

## Environment Variables

- `LANGUAGE`: Target programming language (java, python, go, c) - **Required**
- `RUNTIME_API_KEY`: API key for remote workspace evaluation
- `RUNTIME_API_URL`: Runtime API URL (default: https://runtime.eval.all-hands.dev)
- `SDK_SHORT_SHA`: Override SDK SHA for image selection

### Comparing Docker vs Remote Workspace

| Aspect | Docker Workspace | Remote Workspace |
|--------|-----------------|------------------|
| **Setup** | Simple, no prerequisites | Requires pre-built images + API key |
| **Scale** | Limited by local resources | Hundreds of parallel workers |
| **Speed** | Slower for large evaluations | Much faster with parallelization |
| **Cost** | Local compute only | API usage costs |
| **Use Case** | Development, testing | Production benchmarks, research |

## Evaluation

After running inference (with either workspace type), evaluate the generated patches using the Multi-SWE-Bench evaluation:

**Basic evaluation:**

```bash
uv run multi-swebench-eval output.jsonl
```

**Advanced options:**

```bash
# Specify custom dataset and output file
uv run multi-swebench-eval output.jsonl \
  --dataset bytedance-research/Multi-SWE-Bench \
  --output-file results.multi-swebench.jsonl

# Only convert format without running evaluation
uv run multi-swebench-eval output.jsonl --skip-evaluation
```

The evaluation script will:
1. Convert OpenHands output format to Multi-SWE-Bench prediction format
2. Run the Multi-SWE-Bench evaluation harness (unless `--skip-evaluation` is used)
3. Report pass/fail results for each instance

## Examples

### Java Example
```bash
# Test on a small subset first
echo "apache__commons-cli__CLI-291" > test_instances.txt
echo "google__gson__Gson-1043" >> test_instances.txt

LANGUAGE=java uv run multi-swebench-infer .llm_config/sonnet-4-5.json \
    --select test_instances.txt \
    --workspace remote \
    --num-workers 2 \
    --max-iterations 300
```

### Python Example
```bash
# Full-scale Python evaluation
LANGUAGE=python uv run multi-swebench-infer .llm_config/sonnet-4-5.json \
    --dataset bytedance-research/Multi-SWE-Bench \
    --split python_verified \
    --workspace remote \
    --num-workers 32 \
    --max-iterations 500
```

## Migration from Legacy Structure

This benchmark has been migrated from the legacy `workspace/multi_swe_bench_old` structure to follow the new OpenHands benchmarks SDK pattern. Key changes include:

- Updated import paths to use `benchmarks.utils.*`
- Modernized evaluation framework using SDK components
- Improved Docker image handling
- Enhanced multi-language support
- Standardized CLI interface

## Troubleshooting

### Common Issues

**Error: "RUNTIME_API_KEY environment variable is not set"**
- Solution: Export the `RUNTIME_API_KEY` environment variable before running

**Error: "LANGUAGE environment variable is not set"**
- Solution: Set the `LANGUAGE` environment variable to one of: java, python, go, c

**Error: "Agent server image ... does not exist in container registry"**
- Solution: Ensure images are pre-built and pushed using the build_images script
- Verify the SDK SHA matches between your local submodule and the built images
- Check that images are publicly accessible in the registry

**Error: "Connection timeout" or API errors**
- Solution: Check your network connectivity
- Verify the `RUNTIME_API_URL` is correct
- Ensure your API key has sufficient quota for the number of workers

### Debug Mode

Enable debug logging:
```bash
export OPENHANDS_LOG_LEVEL=DEBUG
LANGUAGE=java uv run multi-swebench-infer --help
```

## References

- [Multi-SWE-Bench Paper](https://arxiv.org/abs/2408.14354)
- [Multi-SWE-Bench GitHub](https://github.com/bytedance-research/Multi-SWE-Bench)
- [SWE-Bench Original](https://github.com/princeton-nlp/SWE-bench)