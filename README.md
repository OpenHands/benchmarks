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

### 2. Build Docker Images for SWE-Bench Evaluation
Build ALL docker images for SWE-Bench.
```bash
uv run benchmarks/swe_bench/build_images.py \
  --dataset princeton-nlp/SWE-bench_Verified --split test \
  --critic pass \
  --image ghcr.io/all-hands-ai/agent-server --target binary-minimal
```


### 3. Run SWE-Bench Inference
```bash
# Run evaluation with your configured LLM
uv run swebench-infer .llm_config/example.json \
                      --critic pass \
                      --max-iterations 10 \
                      --n-limit 1
```

### 4. Selecting Specific Instances

You can run evaluation on a specific subset of instances using the `--select` option:

1. Create a text file with one instance ID per line:

**instances.txt:**
```
django__django-11333
astropy__astropy-12345
requests__requests-5555
```

2. Run evaluation with the selection file:
```bash
python -m benchmarks.swe_bench.run_infer \
    --agent-cls CodeActAgent \
    --llm-config llm_config.toml \
    --max-iterations 30 \
    --select instances.txt \
    --eval-output-dir ./evaluation_results
```

This will only evaluate the instances listed in the file.

### 5. Evaluate SWE-Bench Results
After running inference, evaluate the results using the official SWE-Bench evaluation:

```bash
# Convert output format and run SWE-Bench evaluation
uv run swebench-eval output.jsonl

# Or specify custom dataset and output file
uv run swebench-eval output.jsonl --dataset princeton-nlp/SWE-bench_Lite --output-file results.swebench.jsonl

# Only convert format without running evaluation
uv run swebench-eval output.jsonl --skip-evaluation

# Install SWE-Bench if needed
uv run swebench-eval output.jsonl --install-swebench
```

The script will:
1. Convert OpenHands output format to SWE-Bench prediction format
2. Install SWE-Bench if not already available (optional)
3. Run the official SWE-Bench evaluation harness

## Links

- **Original OpenHands**: https://github.com/All-Hands-AI/OpenHands/
- **Agent SDK**: https://github.com/All-Hands-AI/agent-sdk
- **SWE-Bench**: https://www.swebench.com/
