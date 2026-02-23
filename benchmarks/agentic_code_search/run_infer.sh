uv run python -m benchmarks.agentic_code_search.run_infer \
    --dataset adityasoni17/SWE-bench_Verified-code-search \
    --llm-config-path .llm_config/qwen3_cso_14b.json \
    --system_prompt_file benchmarks/agentic_code_search/prompts/system_prompt_custom_finish2.j2 \
    --user_prompt_file benchmarks/agentic_code_search/prompts/file_module_custom_finish.j2 \
    --tools terminal \
    --max-iterations 15 \
    --num-workers 16 \
    --output-dir ./qwen3_cso_14b_gspo \
    --n-limit 500 \
    --runtime local \
    --workspace_base_dir /tmp/testbed/

uv run python -m benchmarks.agentic_code_search.run_infer \
    --dataset adityasoni17/SWE-bench_Lite-code-search \
    --llm-config-path .llm_config/qwen3_4b_instruct.json \
    --system_prompt_file benchmarks/agentic_code_search/prompts/system_prompt_custom_finish.j2 \
    --user_prompt_file benchmarks/agentic_code_search/prompts/file_module_custom_finish.j2 \
    --tools terminal localization_finish \
    --max-iterations 15 \
    --num-workers 16 \
    --output-dir ./qwen3_4b_instruct \
    --n-limit 500 \
    --runtime local \
    --workspace_base_dir /tmp/testbed/

uv run python -m benchmarks.agentic_code_search.run_infer \
    --dataset adityasoni17/SWE-bench_Lite-code-search \
    --llm-config-path .llm_config/qwen3_14b_instruct_openpipe.json \
    --system_prompt_file benchmarks/agentic_code_search/prompts/system_prompt_custom_finish2.j2 \
    --user_prompt_file benchmarks/agentic_code_search/prompts/file_module_custom_finish.j2 \
    --tools terminal localization_finish \
    --max-iterations 15 \
    --num-workers 16 \
    --output-dir ./qwen3_14b_instruct_openpipe \
    --n-limit 500 \
    --runtime local \
    --workspace_base_dir /tmp/testbed/
rm -rf /tmp/testbed/
uv run python -m benchmarks.agentic_code_search.run_infer \
    --dataset adityasoni17/SWE-bench_Pro-code-search \
    --llm-config-path .llm_config/qwen3_14b_instruct_openpipe.json \
    --system_prompt_file benchmarks/agentic_code_search/prompts/system_prompt_custom_finish2.j2 \
    --user_prompt_file benchmarks/agentic_code_search/prompts/file_module_custom_finish.j2 \
    --tools terminal localization_finish \
    --max-iterations 15 \
    --num-workers 16 \
    --output-dir ./qwen3_14b_instruct_openpipe \
    --n-limit 500 \
    --runtime local \
    --workspace_base_dir /tmp/testbed/
rm -rf /tmp/testbed/
uv run python -m benchmarks.agentic_code_search.run_infer \
    --dataset adityasoni17/SWE-bench_Verified-code-search \
    --llm-config-path .llm_config/qwen3_14b_instruct_openpipe.json \
    --system_prompt_file benchmarks/agentic_code_search/prompts/system_prompt_custom_finish2.j2 \
    --user_prompt_file benchmarks/agentic_code_search/prompts/file_module_custom_finish.j2 \
    --tools terminal localization_finish \
    --max-iterations 15 \
    --num-workers 16 \
    --output-dir ./qwen3_14b_instruct_openpipe \
    --n-limit 500 \
    --runtime local \
    --workspace_base_dir /tmp/testbed/
rm -rf /tmp/testbed/


uv run python -m benchmarks.agentic_code_search.run_infer \
    --dataset adityasoni17/SWE-bench_Lite-code-search \
    --llm-config-path .llm_config/qwen3_cso_14b_str_op.json \
    --system_prompt_file benchmarks/agentic_code_search/prompts/system_prompt.j2 \
    --user_prompt_file benchmarks/agentic_code_search/prompts/file_module_short.j2 \
    --tools terminal \
    --enable_sdk_default_tools \
    --max-iterations 15 \
    --num-workers 16 \
    --output-dir ./qwen3_cso_14b_str_op \
    --n-limit 500 \
    --runtime local \
    --workspace_base_dir /tmp/testbed/
rm -rf /tmp/testbed/
uv run python -m benchmarks.agentic_code_search.run_infer \
    --dataset adityasoni17/SWE-bench_Pro-code-search \
    --llm-config-path .llm_config/qwen3_cso_14b_str_op.json \
    --system_prompt_file benchmarks/agentic_code_search/prompts/system_prompt.j2 \
    --user_prompt_file benchmarks/agentic_code_search/prompts/file_module_short.j2 \
    --tools terminal \
    --enable_sdk_default_tools \
    --max-iterations 15 \
    --num-workers 16 \
    --output-dir ./qwen3_cso_14b_str_op \
    --n-limit 500 \
    --runtime local \
    --workspace_base_dir /tmp/testbed/
rm -rf /tmp/testbed/
uv run python -m benchmarks.agentic_code_search.run_infer \
    --dataset adityasoni17/SWE-bench_Verified-code-search \
    --llm-config-path .llm_config/qwen3_cso_14b_str_op.json \
    --system_prompt_file benchmarks/agentic_code_search/prompts/system_prompt.j2 \
    --user_prompt_file benchmarks/agentic_code_search/prompts/file_module_short.j2 \
    --tools terminal \
    --enable_sdk_default_tools \
    --max-iterations 15 \
    --num-workers 16 \
    --output-dir ./qwen3_cso_14b_str_op \
    --n-limit 500 \
    --runtime local \
    --workspace_base_dir /tmp/testbed/
rm -rf /tmp/testbed/