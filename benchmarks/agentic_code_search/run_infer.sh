uv run python -m benchmarks.agentic_code_search.run_infer \
    --dataset_file train.parquet \
    --llm-config-path .llm_config/qwen3.json \
    --max-iterations 25 \
    --num-workers 8 \
    --output-dir ./docker_outputs \
    --n-limit 10 \
    --runtime docker

uv run python -m benchmarks.agentic_code_search.run_infer \
    --dataset_file train.parquet \
    --llm-config-path .llm_config/qwen3.json \
    --max-iterations 25 \
    --num-workers 1 \
    --output-dir ./local_outputs \
    --n-limit 1 \
    --workspace_base_dir /tmp/workspace/ \
    --runtime local