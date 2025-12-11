# Commit0 Benchmark Evaluation

This directory contains the implementation for running Commit0 evaluation using OpenHands agents.

## Overview

Commit0 is a benchmark for evaluating AI agents on completing Python function implementations. The benchmark tests an agent's ability to understand function stubs, implement the required functionality, and pass unit tests.

## Dataset

- **Source**: wentingzhao/commit0_combined
- **Splits**: 
  - `lite` - Smaller curated subset of repositories
  - `all` - Full dataset with all repositories
  - Individual repository names can also be specified

## Usage

### Run Inference

Run evaluation using the console script (recommended):

```bash
uv run commit0-infer \
    path/to/llm_config.json \
    --dataset wentingzhao/commit0_combined \
    --split test \
    --repo-split lite \
    --max-iterations 100
```

Or run directly as a module:

```bash
uv run python -m benchmarks.commit0.run_infer \
    path/to/llm_config.json \
    --dataset wentingzhao/commit0_combined \
    --split test \
    --repo-split lite \
    --max-iterations 100
```

**Key Arguments:**

- `--repo-split`: Choose between `lite`, `all`, or a specific repository name
- `--max-iterations`: Maximum number of agent iterations per instance
- `--num-workers`: Number of parallel workers for evaluation
- `--prompt-path`: Path to custom prompt template (default: `prompts/default.j2`)

**Selecting specific instances:**

You can run evaluation on a specific subset by creating a text file with instance IDs:

```bash
# Create instances.txt with one instance ID per line
echo "repo_name_1" > instances.txt
echo "repo_name_2" >> instances.txt

# Run with selection
uv run commit0-infer \
    path/to/llm_config.json \
    --select instances.txt
```

### Output Structure

The evaluation will create the following directory structure:

```
eval_outputs/
└── <dataset_name>-<repo_split>/
    ├── output.jsonl              # Main results file
    └── repos/
        └── <repo_name>/
            ├── <repo_name>.zip              # Workspace snapshot
            ├── <repo_name>_patch.diff       # Generated git patch
            ├── <repo_name>_test_output.txt  # Test execution output
            └── <repo_name>_pytest_exit_code.txt  # Pytest exit code
```

### Evaluation Results

Each line in `output.jsonl` contains:

- `instance_id`: Repository name
- `test_result`: Dictionary with evaluation metrics including:
  - `name`: Repository name
  - `sum`: Total test runtime
  - `passed`: Ratio of passed tests
  - `num_passed`: Number of passed tests
  - `num_tests`: Total number of tests
- `instruction`: The prompt given to the agent
- `history`: Complete interaction history
- `error`: Any errors encountered (if applicable)

## Docker Images

The evaluation uses pre-built Docker images from:
- Default prefix: `docker.io/wentingzhao/`
- Image naming: `<repo_name>:v0`

You can override the prefix using the `EVAL_DOCKER_IMAGE_PREFIX` environment variable:

```bash
EVAL_DOCKER_IMAGE_PREFIX=docker.io/myprefix/ uv run commit0-infer ...
```

## References

- [Commit0 GitHub Repository](https://github.com/commit-0/commit0)
