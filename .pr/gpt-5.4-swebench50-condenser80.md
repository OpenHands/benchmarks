# GPT-5.4 SWE-bench 50-instance eval branch

This branch temporarily changes the SWE-bench condenser default for a manual evaluation run.

## Intended runtime behavior
- benchmark: `swebench`
- enable_condenser: `true`
- condenser_max_size: `80`
- condenser_keep_first: `2`
- scope: `benchmarks/swebench/config.py` only

## Notes
- This branch is intended for evaluation only, not for merge as-is.
- The matching SDK branch is `eval/gpt-5.4-swebench50-condenser80`.
