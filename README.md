# OpenHands Benchmarks Migration

⚠️ **Migration in Progress**: We are currently migrating the benchmarks infrastructure from [OpenHands](https://github.com/All-Hands-AI/OpenHands/) to work with the [OpenHands Agent SDK](https://github.com/All-Hands-AI/agent-sdk).

## Prerequisites

Before running any benchmarks, you need to set up the environment and ensure the local Agent SDK submodule is initialized.

```bash
make build
```

<details>
<summary>📦 Submodule & Environment Setup (click to expand)</summary>

### 🧩 1. Initialize the Agent SDK submodule

The Benchmarks project uses a **local git submodule** for the [OpenHands Agent SDK](https://github.com/All-Hands-AI/agent-sdk).  
This ensures your code runs against a specific, reproducible commit.

Run once after cloning (already done in `make build` for you):

```bash
git submodule update --init --recursive
```

This command will:
- clone the SDK into `vendor/agent-sdk/`
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
cd vendor/agent-sdk
git fetch
git checkout <new_commit_or_branch>
cd ../..
git add vendor/agent-sdk
git commit -m "Update agent-sdk submodule to <new_commit_sha>"
```

Then re-run:

```bash
make build
```

to rebuild your environment with the new SDK code.

</details>

## Available Benchmarks

This repository contains multiple benchmarks:

- **SWE-Bench**: Software engineering tasks requiring code changes and bug fixes
- **OpenAgentSafety**: AI agent safety evaluation in workplace scenarios with NPC interactions

## Quick Start

### 1. Configure Your LLM

Define your LLM config as a JSON following the model fields type in the [LLM class](https://github.com/All-Hands-AI/agent-sdk/blob/main/openhands/sdk/llm/llm.py#L93), [for example](.llm_config/example.json), you can write the following to `.llm_config/example.json`:

```json
{
  "model": "litellm_proxy/anthropic/claude-sonnet-4-20250514",
  "base_url": "https://llm-proxy.eval.all-hands.dev",
  "api_key": "YOUR_API_KEY_HERE"
}
```

You may validate the correctness of your config by running `uv run validate-cfg .llm_config/YOUR_CONFIG_PATH.json`

## SWE-Bench Evaluation

### 2. Build Docker Images for SWE-Bench
Build ALL docker images for SWE-Bench.
```bash
uv run benchmarks/swe_bench/build_images.py \
  --dataset princeton-nlp/SWE-bench_Verified --split test \
  --image ghcr.io/all-hands-ai/agent-server --target binary-minimal
```

### 3. Run SWE-Bench Evaluation
```bash
# Run evaluation with your configured LLM
uv run swebench-infer .llm_config/sonnet-4.json
```

## OpenAgentSafety Evaluation

OpenAgentSafety uses the same LLM config approach as SWE-Bench, but requires additional setup for NPC interactions and TheAgentCompany services.

### 4. Start Docker and Set Up Services

OpenAgentSafety requires TheAgentCompany infrastructure (GitLab, ownCloud, RocketChat, etc.) for realistic NPC interactions:

```bash
# 1. Start Docker daemon
sudo dockerd > /tmp/docker.log 2>&1 &
sleep 5

# 2. Verify Docker is running
sudo docker version

# 3. Fix Docker socket permissions
sudo chmod 666 /var/run/docker.sock

# 4. Set up TheAgentCompany services (~30GB download, 15-30 min setup)
cd /workspace/project
curl -fsSL https://github.com/TheAgentCompany/the-agent-company-backup-data/releases/download/setup-script-20241208/setup.sh | sh

# 5. Set NPC environment variables
export NPC_API_KEY="your-openai-api-key"
export NPC_BASE_URL="https://api.openai.com/v1"
export NPC_MODEL="gpt-4o-mini"
```

### 5. Run OpenAgentSafety Evaluation
```bash
cd benchmarks
uv run openagentsafety-infer .llm_config/gpt-4o-mini.json \
  --dataset mgulavani/openagentsafety_full_updated_v3 \
  --split train \
  --output-dir ./results \
  --num-workers 1 \
  --n-limit 2 \
  --critic pass
```

**Note**: The setup requires ~30GB disk space and sufficient system resources for multiple services (GitLab, ownCloud, RocketChat, databases).

See [benchmarks/openagentsafety/README.md](benchmarks/openagentsafety/README.md) for detailed documentation.

## Selecting Specific Instances

Both SWE-Bench and OpenAgentSafety support running evaluation on a specific subset of instances using the `--select` option:

1. Create a text file with one instance ID per line:

**instances.txt:**
```
django__django-11333
astropy__astropy-12345
requests__requests-5555
```

2. Run evaluation with the selection file:

**For SWE-Bench:**
```bash
python -m benchmarks.swe_bench.run_infer \
    --agent-cls CodeActAgent \
    --llm-config llm_config.toml \
    --max-iterations 30 \
    --select instances.txt \
    --eval-output-dir ./evaluation_results
```

**For OpenAgentSafety:**
```bash
uv run openagentsafety-infer .llm_config/gpt-4o-mini.json \
  --dataset mgulavani/openagentsafety_full_updated_v3 \
  --split train \
  --output-dir ./results \
  --select instances.txt \
  --critic pass
```

This will only evaluate the instances listed in the file.

## Links

- **Original OpenHands**: https://github.com/All-Hands-AI/OpenHands/
- **Agent SDK**: https://github.com/All-Hands-AI/agent-sdk
- **SWE-Bench**: https://www.swebench.com/