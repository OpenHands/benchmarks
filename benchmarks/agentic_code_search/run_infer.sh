uv run python -m benchmarks.agentic_code_search.run_infer \
    --dataset_file gt_location.jsonl \
    --llm-config-path .llm_config/qwen3.json \
    --max-iterations 25 \
    --num-workers 1 \
    --output-dir ./qwen3_8b_benchmarking \
    --n-limit 500 \
    --runtime docker