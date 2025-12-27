uv run python -m benchmarks.agentic_code_search.run_infer \
    --dataset_file gt_location.jsonl \
    --llm-config-path .llm_config/example.json \
    --max-iterations 5 \
    --num-workers 1 \
    --output-dir ./agentic_code_search_outputs \
    --n-limit 1 \
    --runtime local