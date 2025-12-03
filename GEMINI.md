# OpenHands Benchmarks

## Project Overview

This repository contains the benchmark evaluation infrastructure for [OpenHands](https://github.com/OpenHands/OpenHands/) agents. It provides standardized pipelines to test agent capabilities across various domains, including software engineering, general reasoning, and safety.

The project is built around the [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk), utilizing it as a submodule to ensure reproducible evaluations against specific SDK versions.

## Getting Started

### Prerequisites

-   **Python**: >= 3.12
-   **uv**: >= 0.8.13 (High-performance Python package installer)
-   **Docker**: Required for local workspace evaluation.

### Setup

1.  **Initialize Environment**:
    The project uses a `Makefile` to handle submodule initialization and dependency installation via `uv`.

    ```bash
    make build
    ```
    This command will:
    -   Update `vendor/software-agent-sdk` submodule.
    -   Install dependencies defined in `pyproject.toml`.
    -   Install pre-commit hooks.

2.  **Configure LLM**:
    Create a JSON configuration file in `.llm_config/` (e.g., `.llm_config/my_config.json`).

    ```json
    {
      "model": "litellm_proxy/anthropic/claude-sonnet-4-20250514",
      "base_url": "https://llm-proxy.eval.all-hands.dev",
      "api_key": "YOUR_API_KEY_HERE"
    }
    ```

    Validate it:
    ```bash
    uv run validate-cfg .llm_config/my_config.json
    ```

## Available Benchmarks

The benchmarks are located in the `benchmarks/` directory. Each has its own specific usage patterns, but generally follows the `uv run <benchmark>-infer` pattern.

### 1. SWE-Bench (`benchmarks/swe_bench/`)
Evaluates agents on real-world software engineering tasks (GitHub issues).

-   **Key Command**: `swebench-infer`
-   **Usage**:
    ```bash
    # Local Docker execution
    uv run swebench-infer path/to/config.json --dataset princeton-nlp/SWE-bench_Lite --split test --workspace docker
    ```
-   **Remote Execution**: Supports scalable cloud evaluation via `--workspace remote`.

### 2. GAIA (`benchmarks/gaia/`)
Evaluates general AI assistants on multi-step reasoning, web search, and tool use.

-   **Key Command**: `gaia-infer`
-   **Usage**:
    ```bash
    # Requires TAVILY_API_KEY
    TAVILY_API_KEY=... uv run gaia-infer path/to/config.json --level 2023_level1 --split validation
    ```

### 3. OpenAgentSafety (`benchmarks/openagentsafety/`)
Evaluates agent safety in workplace scenarios involving interactions with NPCs.

-   **Key Command**: `openagentsafety-infer`
-   **Usage**:
    ```bash
    uv run openagentsafety-infer path/to/config.json --dataset mgulavani/openagentsafety_full_updated_v3 --split train
    ```
-   **Requirements**: Requires setting up TheAgentCompany services (Docker) and specific environment variables (`NPC_API_KEY`, etc.).

### 4. SWT-Bench (`benchmarks/swt_bench/`)
Likely a variant or specific subset for software tasks (details in `benchmarks/swt_bench/README.md`).

-   **Key Command**: `swtbench-infer`
-   **Usage**:
    ```bash
    uv run swtbench-infer path/to/config.json --critic pass
    ```

## Development Workflow

### Code Quality
The project uses `ruff` for linting and formatting.

-   **Format**: `make format`
-   **Lint**: `make lint`
-   **Clean**: `make clean` (removes cache files)

### Managing the SDK Submodule
The `vendor/software-agent-sdk` is a git submodule.

-   **Update SDK**:
    ```bash
    cd vendor/software-agent-sdk
    git fetch && git checkout <new_commit>
    cd ../..
    git add vendor/software-agent-sdk
    git commit -m "Update SDK..."
    make build
    ```

## Key Files & Directories

-   `pyproject.toml`: Defines dependencies and CLI entry points (`project.scripts`).
-   `Makefile`: automation for build, lint, and format tasks.
-   `benchmarks/`: Source code for all benchmarks.
-   `vendor/`: Contains the `software-agent-sdk` submodule.
-   `.llm_config/`: Directory for storing LLM connection details (gitignored).
