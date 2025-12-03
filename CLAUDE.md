# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenHands Benchmarks is a benchmark evaluation infrastructure for testing [OpenHands](https://github.com/OpenHands/OpenHands/) agents. It provides standardized evaluation pipelines across various real-world tasks. The project uses the [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk) (included as a git submodule in `vendor/software-agent-sdk/`).

## Common Commands

```bash
# Initial setup (initialize submodule + install dependencies + pre-commit hooks)
make build

# Format code
make format

# Lint code (with auto-fix)
make lint

# Run tests
uv run pytest tests/

# Run a single test
uv run pytest tests/test_metrics.py::test_specific_function

# Validate LLM config
uv run validate-cfg .llm_config/your_config.json

# Run SWE-Bench inference (local Docker)
uv run swebench-infer .llm_config/config.json --dataset princeton-nlp/SWE-bench_Lite --split test --workspace docker

# Run SWE-Bench inference (remote, requires RUNTIME_API_KEY)
uv run swebench-infer .llm_config/config.json --workspace remote --num-workers 32

# Run GAIA inference (requires TAVILY_API_KEY)
TAVILY_API_KEY=xxx uv run gaia-infer .llm_config/config.json --level 2023_level1 --split validation

# Run SWT-Bench inference
uv run swtbench-infer .llm_config/config.json --critic pass

# Evaluate SWE-Bench results
uv run swebench-eval output.jsonl
```

## Architecture

### Directory Structure

- `benchmarks/` - Core benchmark implementations
  - `swe_bench/` - SWE-Bench (GitHub issues → code patches)
  - `gaia/` - GAIA (multi-step reasoning tasks)
  - `swt_bench/` - SWT-Bench (test generation)
  - `openagentsafety/` - AI safety evaluation
  - `utils/` - Shared evaluation infrastructure
- `vendor/software-agent-sdk/` - OpenHands SDK submodule (includes `openhands-sdk`, `openhands-tools`, `openhands-workspace`, `openhands-agent-server`)

### Evaluation Framework

The evaluation framework follows an abstract orchestrator pattern:

1. **`Evaluation`** (`benchmarks/utils/evaluation.py`) - Abstract base class with process-based parallelization
   - Subclasses implement: `prepare_instances()`, `prepare_workspace()`, `evaluate_instance()`
   - Supports iterative mode with multiple attempts (configurable via `max_attempts`)
   - Handles retry logic, conversation archiving, and multiprocess logging

2. **`EvalMetadata`** (`benchmarks/utils/models.py`) - Configuration model for evaluation runs
   - LLM config, dataset info, max iterations, workspace type, critic settings

3. **`EvalInstance`** / **`EvalOutput`** - Input/output data models

4. **Critics** (`benchmarks/utils/critics.py`) - Determine success/failure for retry logic

### Workspace Types

- **Docker Workspace**: Local containers, images built on-demand
- **Remote Workspace**: Cloud-based via runtime API (`RUNTIME_API_KEY` required), enables massive parallelization

### Key Patterns

- Benchmark commands are registered as scripts in `pyproject.toml` under `[project.scripts]`
- Prompt templates use Jinja2 (`.j2` files in each benchmark's `prompts/` directory)
- Results written to JSONL files with per-instance conversation archives (`.tar.gz`)
- Workflow memory system supports: `none`, `offline_no_retrieve`, `offline_retrieve`, `online` modes

## Pre-commit Hooks

Commits run: Ruff format → Ruff lint → pycodestyle → Pyright type checking

## Environment Variables

- `RUNTIME_API_KEY` - Required for remote workspace
- `RUNTIME_API_URL` - Override runtime API endpoint (default: `https://runtime.eval.all-hands.dev`)
- `SDK_SHORT_SHA` - Override SDK SHA for image selection
- `TAVILY_API_KEY` - Required for GAIA benchmark (web search)
- `SKIP_BUILD` - Set to `1` to skip Docker image building (use pre-built images)
