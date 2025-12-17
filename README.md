# OpenHands Benchmarks

This repository contains benchmark evaluation infrastructure for [OpenHands](https://github.com/OpenHands/OpenHands/) agents. It provides standardized evaluation pipelines for testing agent capabilities across various real-world tasks.

⚠️ **Migration in Progress**: We are currently migrating the [benchmarks from OpenHands V0](https://github.com/OpenHands/OpenHands/tree/main/evaluation) to work with the [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk) infrastructure in V1.

## Available Benchmarks

| Benchmark | Description | Status |
|-----------|-------------|--------|
| [SWE-Bench](benchmarks/swebench/) | Software engineering tasks from GitHub issues | ✅ Active |
| [GAIA](benchmarks/gaia/) | General AI assistant tasks requiring multi-step reasoning | ✅ Active |
| [Commit0](benchmarks/commit0/) | Python function implementation tasks with unit tests | ✅ Active |
| [OpenAgentSafety](benchmarks/openagentsafety/) | AI agent safety evaluation in workplace scenarios with NPC interactions | ✅ Active |

See the individual benchmark directories for detailed usage instructions.

## Quick Start

### Prerequisites

Before running any benchmarks, you need to set up the environment and ensure the local Agent SDK submodule is initialized.

```bash
make build
```

<details>
<summary>📦 Submodule & Environment Setup (click to expand)</summary>

### 🧩 1. Initialize the Agent SDK submodule

The Benchmarks project uses a **local git submodule** for the [OpenHands Agent SDK](https://github.com/OpenHands/software-agent-sdk).
This ensures your code runs against a specific, reproducible commit.

Run once after cloning (already done in `make build` for you):

```bash
git submodule update --init --recursive
```

This command will:
- clone the SDK into `vendor/software-agent-sdk/`
- check out the exact commit pinned by this repo
- make it available for local development (`uv sync` will install from the local folder)

If you ever clone this repository again, remember to re-initialize the submodule with the same command.

---

### 🏗️ 2. Build the environment

Once the submodule is set up, install dependencies via [uv](https://docs.astral.sh/uv):

```bash
make build
```

This runs:

```bash
uv sync
```

and ensures the `openhands-*` packages (SDK, tools, workspace, agent-server) are installed **from the local workspace** declared in `pyproject.toml`.

---

### 🔄 3. Update the submodule (when SDK changes)

If you want to update to a newer version of the SDK:

```bash
cd vendor/software-agent-sdk
git fetch
git checkout <new_commit_or_branch>
cd ../..
git add vendor/software-agent-sdk
git commit -m "Update software-agent-sdk submodule to <new_commit_sha>"
```

Then re-run:

```bash
make build
```

to rebuild your environment with the new SDK code.

</details>

### Configure Your LLM

All benchmarks require an LLM configuration file. Define your LLM config as a JSON following the model fields in the [LLM class](https://github.com/OpenHands/software-agent-sdk/blob/main/openhands/sdk/llm/llm.py#L93).

**Example** (`.llm_config/example.json`):

```json
{
  "model": "litellm_proxy/anthropic/claude-sonnet-4-20250514",
  "base_url": "https://llm-proxy.eval.all-hands.dev",
  "api_key": "YOUR_API_KEY_HERE"
}
```

Validate your configuration:

```bash
uv run validate-cfg .llm_config/YOUR_CONFIG_PATH.json
```

## Running Benchmarks

After setting up the environment and configuring your LLM, see the individual benchmark directories for specific usage instructions:

- **[SWE-Bench](benchmarks/swebench/)**: Software engineering tasks from GitHub issues
- **[GAIA](benchmarks/gaia/)**: General AI assistant tasks requiring multi-step reasoning  
- **[OpenAgentSafety](benchmarks/openagentsafety/)**: AI agent safety evaluation in workplace scenarios with NPC interactions

## Workspace Types

Benchmarks support two workspace types for running evaluations:

### Docker Workspace (Default)

Uses local Docker containers to run agent evaluations. Images are built locally on-demand.

- **Pros**: No additional setup required, works offline
- **Cons**: Resource-intensive on local machine, slower for large-scale evaluations
- **Use case**: Development, testing, small-scale evaluations

### Remote Workspace

Uses a [remote runtime API](https://openhands.dev/blog/evaluation-of-llms-as-coding-agents-on-swe-bench-at-30x-speed) to provision containers in a cloud environment, enabling massive parallelization.

- **Pros**: Scalable to hundreds of parallel workers, no local resource constraints
- **Cons**: Requires pre-built images and API access
- **Use case**: Large-scale evaluations, benchmarking runs

#### How Remote Runtime Works

1. **Pre-build Agent Images**: Agent-server images must be pre-built for a specific SDK commit (SHA) and pushed to a public container registry (e.g., `ghcr.io/openhands/eval-agent-server`)
   
2. **Runtime API**: The remote workspace connects to a runtime API service (default: `https://runtime.eval.all-hands.dev`) that provisions containers on-demand

3. **Image Resolution**: Before starting evaluation, the system verifies that the required image exists in the registry with the correct tag format: `{IMAGE}:{SDK_SHA}-{CUSTOM_TAG}{SUFFIX}`

4. **Parallel Execution**: Each evaluation instance runs in its own isolated container, allowing for massive parallelization (e.g., 32+ concurrent workers)

#### Prerequisites for Remote Workspace

1. **Pre-built Images**: Images must be built and pushed to a public registry
   - In this repository, add one of the following labels to a PR to trigger image builds:
     - `build-swebench-50`: Build 50 images (quick testing)
     - `build-swebench-200`: Build 200 images (medium testing)
     - `build-swebench`: Build all images (full evaluation)
   - Images are tagged with the SDK SHA from the `vendor/software-agent-sdk` submodule

2. **Runtime API Key**: Set the `RUNTIME_API_KEY` environment variable
   ```bash
   export RUNTIME_API_KEY="your-api-key-here"
   ```

3. **Optional Configuration**:
   - `RUNTIME_API_URL`: Override the default API endpoint (default: `https://runtime.eval.all-hands.dev`)
   - `SDK_SHORT_SHA`: Override the SDK SHA for image selection (default: auto-detected from submodule)

See individual benchmark READMEs for specific usage examples.

## SDK Compatibility and Version Management

⚠️ **Important**: The benchmarks repository depends on the [OpenHands Agent SDK](https://github.com/OpenHands/software-agent-sdk), and **not every version of the benchmarks is compatible with every version of the SDK**. As the SDK evolves and introduces new features, the benchmarks code may adopt these features, creating version dependencies.

### SWE-Bench image layering (docutils/roman)

Some SWE-Bench instances (notably `sphinx-doc`) require `docutils<0.21` and `roman`. The build pipeline now wraps only those images that need the extra layer:
- `benchmarks/swebench/build_images.py` wraps images for repos in a small allowlist (currently `sphinx-doc`).
- Other repos (e.g., scikit-learn) keep the base image unchanged.
- `run_infer.py` automatically picks the wrapped tag for allowlisted repos and the base tag for the rest.

When running or dispatching builds, no extra flags are needed—the selective wrapping is handled for you.

### Evaluating Different SDK Versions

When evaluating a specific SDK version, you need to ensure the benchmarks code is compatible with that SDK version. You have two options:

1. **Use the `benchmarks-commit` parameter in the workflow** (Recommended):
   - When manually triggering the `build-swe-bench-images` workflow, specify both:
     - `sdk-commit`: The SDK version you want to evaluate
     - `benchmarks-commit`: A benchmarks commit that's compatible with that SDK version
   
2. **Manually check out compatible versions locally**:
   ```bash
   # Check out a benchmarks commit that's compatible with your target SDK version
   git checkout <benchmarks-commit>
   
   # Update the SDK submodule to your target version
   cd vendor/software-agent-sdk
   git checkout <sdk-commit>
   cd ../..
   
   # Rebuild the environment
   make build
   ```

### Example: SDK Critic Module

A notable example of version dependency is the SDK critic module. As of SDK commit [`79868ae5`](https://github.com/OpenHands/software-agent-sdk/commit/79868ae5) (November 17, 2025), the OpenHands Agent SDK introduced the `openhands.sdk.critic` module. Current benchmarks code imports `CriticBase` from this module, which means:

- **SDK versions ≥ `79868ae5`**: Compatible with current benchmarks code
- **SDK versions < `79868ae5`**: Require an older benchmarks commit (before the critic import was added)

To check if a specific benchmarks commit requires the critic module:
```bash
git show <commit>:benchmarks/utils/models.py | grep "from openhands.sdk.critic"
```

If this command returns output, that benchmarks commit requires an SDK version with the critic module.

## Links

- **Original OpenHands**: https://github.com/OpenHands/OpenHands/
- **Agent SDK**: https://github.com/OpenHands/software-agent-sdk
- **SWE-Bench**: https://www.swebench.com/
