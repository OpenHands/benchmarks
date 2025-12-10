uv run swebench-infer .llm_config/moonshot_kimi-k2-0905.json \
    --dataset princeton-nlp/SWE-bench_Lite \
    --split test \
    --workspace remote \
    --num-workers 12 \
    --max-iterations 100 \
    --n-limit 300 \