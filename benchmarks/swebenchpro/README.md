# SWE-Bench Pro Benchmark Evaluation

This directory contains the OpenHands benchmark integration for [SWE-Bench Pro](https://scale.com/leaderboard/swe_bench_pro_public), a public long-horizon software engineering benchmark released by Scale AI.

## Dataset

- **Dataset**: `ScaleAI/SWE-bench_Pro`
- **Split**: `test`
- **Official harness**: [`scaleapi/SWE-bench_Pro-os`](https://github.com/scaleapi/SWE-bench_Pro-os)
- **Official images**: `jefzda/sweap-images:<dockerhub_tag>`

## Running Inference

SWE-Bench Pro reuses the phased image-build pipeline from `benchmarks/swebench/`, but resolves official base images from each dataset row's `dockerhub_tag` field.

### Build agent-server images

```bash
uv run python -m benchmarks.swebenchpro.build_images \
  --dataset ScaleAI/SWE-bench_Pro \
  --split test \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal
```

### Run inference

```bash
uv run swebenchpro-infer path/to/llm_config.json \
  --dataset ScaleAI/SWE-bench_Pro \
  --split test \
  --workspace docker
```

Remote and apptainer workspaces use the same image tags produced by the phased build pipeline.

## Running Evaluation

The evaluation wrapper converts OpenHands `output.jsonl` files into the official SWE-Bench Pro patch format, downloads a pinned checkout of the official harness on first use, materializes the required dataset rows as JSONL, and then invokes the upstream evaluation script.

```bash
uv run swebenchpro-eval path/to/output.jsonl \
  --dataset ScaleAI/SWE-bench_Pro \
  --split test \
  --use-local-docker
```

Helpful options:

- `--skip-evaluation`: only write the converted patch file.
- `--official-harness-dir <path>`: use an existing local checkout of `scaleapi/SWE-bench_Pro-os` instead of downloading the pinned archive.
- `--no-use-local-docker`: use Modal instead of local Docker.
- `--block-network`: disable network access inside evaluation containers.

The script writes:

- `output.swebenchpro.json`: converted patches in the upstream JSON format.
- `output.report.json`: OpenHands-style evaluation summary with `resolved_ids` for downstream tooling.
- `cost_report.jsonl`: aggregated usage and proxy-cost summary.
