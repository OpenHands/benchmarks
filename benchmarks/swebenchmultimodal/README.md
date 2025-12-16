# SWE-Bench Multimodal

This benchmark implements evaluation for SWE-Bench Multimodal datasets, which include visual elements like images, diagrams, and screenshots alongside the traditional text-based issue descriptions.

## Key Differences from Regular SWE-Bench

1. **Docker Images**: Uses regular SWE-bench docker images (`sweb.eval.*`) as base, with multimodal functionality handled at the application level
2. **Environment Setup**: Skips testbed environment activation (similar to SWE-bench-Live)
3. **Dataset Support**: Designed specifically for `princeton-nlp/SWE-bench_Multimodal` dataset
4. **Multimodal Content**: Handles visual elements (images, diagrams, screenshots) through the agent's multimodal capabilities

## Usage

### Running Inference

```bash
uv run swebenchmultimodal-infer \
  --dataset princeton-nlp/SWE-bench_Multimodal \
  --split test \
  --llm-config .llm_config/your-config.json \
  --output-dir ./output
```

### Running Evaluation

After running inference, you can evaluate the results using:

```bash
uv run swebenchmultimodal-eval <path_to_output.jsonl>
```

This will:
1. Convert the OpenHands output format to SWE-Bench prediction format
2. Run the SWE-Bench Multimodal evaluation with the `--modal true` flag
3. Generate a cost report

Example:
```bash
uv run swebenchmultimodal-eval ./output/output.jsonl --workers 8
```

For more evaluation options:
```bash
uv run swebenchmultimodal-eval --help
```

You can also refer to the official [SWE-bench Multimodal repository](https://github.com/SWE-bench/SWE-bench) for additional evaluation details.

### Building Docker Images

Pre-build all required docker images:

```bash
uv run benchmarks/swebenchmultimodal/build_images.py \
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

1. **Visual Content**: The agent will have access to images and visual elements through the problem statement and workspace
2. **No Testbed**: Unlike regular SWE-Bench, multimodal instances don't use the testbed environment
3. **Docker Images**: Uses regular SWE-bench docker images (`sweb.eval.*`) as the base environment
4. **Multimodal Processing**: Visual content is processed by the agent's multimodal capabilities, not at the container level

## Example

```bash
# Run inference on a small subset
uv run swebenchmultimodal-infer \
  --dataset princeton-nlp/SWE-bench_Multimodal \
  --split test \
  --llm-config .llm_config/claude-3-5-sonnet.json \
  --max-instances 5 \
  --output-dir ./multimodal_output

# For evaluation, see the official SWE-bench multimodal repository
# https://github.com/SWE-bench/SWE-bench
```