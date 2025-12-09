# SWE-Bench Multimodal

This benchmark implements evaluation for SWE-Bench Multimodal datasets, which include visual elements like images, diagrams, and screenshots alongside the traditional text-based issue descriptions.

## Key Differences from Regular SWE-Bench

1. **Docker Images**: Uses multimodal-specific docker images with `sweb.mm.eval` prefix instead of `sweb.eval`
2. **Environment Setup**: Skips testbed environment activation (similar to SWE-bench-Live)
3. **Dataset Support**: Designed specifically for `princeton-nlp/SWE-bench_Multimodal` dataset

## Usage

### Running Inference

```bash
uv run swebench-multimodal-infer \
  --dataset princeton-nlp/SWE-bench_Multimodal \
  --split test \
  --llm-config .llm_config/your-config.json \
  --output-dir ./output
```

### Running Evaluation

```bash
uv run swebench-multimodal-eval ./output/princeton-nlp__SWE-bench_Multimodal-test/your-model/output.jsonl
```

### Building Docker Images

Pre-build all required docker images:

```bash
uv run benchmarks/swe_bench_multimodal/build_images.py \
  --dataset princeton-nlp/SWE-bench_Multimodal \
  --split test \
  --image ghcr.io/openhands/eval-agent-server
```

## Configuration

The benchmark uses the same configuration options as regular SWE-Bench:

- `--dataset`: Dataset name (should be `princeton-nlp/SWE-bench_Multimodal`)
- `--split`: Dataset split (e.g., `test`, `dev`)
- `--llm-config`: Path to LLM configuration file
- `--max-iterations`: Maximum number of agent iterations
- `--workspace-type`: Either `docker` or `remote`
- `--num-workers`: Number of parallel workers

## Environment Variables

- `SKIP_BUILD=1`: Skip building docker images (use pre-built images)
- `RUNTIME_API_KEY`: Required for remote workspace
- `RUNTIME_API_URL`: Runtime API URL (defaults to https://runtime.eval.all-hands.dev)

## Multimodal Considerations

When working with multimodal instances:

1. **Visual Content**: The agent will have access to images and visual elements through the workspace
2. **No Testbed**: Unlike regular SWE-Bench, multimodal instances don't use the testbed environment
3. **Docker Images**: Ensure you have access to the multimodal-specific docker images

## Example

```bash
# Run inference on a small subset
uv run swebench-multimodal-infer \
  --dataset princeton-nlp/SWE-bench_Multimodal \
  --split test \
  --llm-config .llm_config/claude-3-5-sonnet.json \
  --max-instances 5 \
  --output-dir ./multimodal_output

# Evaluate the results
uv run swebench-multimodal-eval ./multimodal_output/princeton-nlp__SWE-bench_Multimodal-test/claude-3-5-sonnet/output.jsonl
```