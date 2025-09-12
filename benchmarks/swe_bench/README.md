# OpenHands Benchmarks

⚠️ **Migration in Progress**: We are currently migrating the benchmarks infrastructure. At the moment, only local mode evaluation using `run_infer.sh` is working.

This repository contains benchmark evaluation tools for OpenHands agents.

## SWE-Bench Evaluation (Local Mode Only)

Currently, only the SWE-Bench evaluation is working in local mode using the `run_infer.sh` script.

### Prerequisites

1. **Environment Setup**: Install Python environment and configure LLM settings
2. **Docker**: Ensure Docker daemon is running with ample disk space (200-500GB)
3. **LLM Configuration**: Set up your LLM config in `config.toml`

### Running SWE-Bench Evaluation

Use the `run_infer.sh` script to run SWE-Bench evaluation locally:

```bash
./swe_bench/scripts/run_infer.sh [model_config] [git-version] [agent] [eval_limit] [max_iter] [num_workers] [dataset] [dataset_split]

# Example - Run 10 instances with GPT-4 and CodeActAgent
./swe_bench/scripts/run_infer.sh llm.eval_gpt4_1106_preview HEAD CodeActAgent 10
```

### Parameters

- `model_config`: LLM config group name from your `config.toml` (required)
- `git-version`: Git commit hash or tag (default: `HEAD`)
- `agent`: Agent name (default: `CodeActAgent`)
- `eval_limit`: Number of instances to evaluate (default: full dataset)
- `max_iter`: Maximum iterations per instance (default: 100)
- `num_workers`: Number of parallel workers (default: 1)
- `dataset`: HuggingFace dataset name (default: `princeton-nlp/SWE-bench_Lite`)
- `dataset_split`: Dataset split (default: `test`)

### Supported Datasets

- `princeton-nlp/SWE-bench_Lite`
- `princeton-nlp/SWE-bench_Verified`
- `princeton-nlp/SWE-bench`
- `princeton-nlp/SWE-bench_Multimodal`

### Output

Results are saved to `eval_out/` directory as `output.jsonl` files containing the evaluation results.

### Environment Variables

```bash
# Use hint text in evaluation (default: false)
export USE_HINT_TEXT=true

# Enable iterative evaluation protocol (default: false)
export ITERATIVE_EVAL_MODE=true

# Skip maximum retries for debugging (default: false)
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true
```

## Status

- ✅ **SWE-Bench Local Evaluation**: Working with `run_infer.sh`
- ❌ **SWE-Gym**: Not currently supported
- ❌ **SWT-Bench**: Not currently supported
- ❌ **Remote Runtime**: Not currently supported

For detailed SWE-Bench documentation, see [swe_bench/README.md](./swe_bench/README.md).