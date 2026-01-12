<ROLE>
You are a collaborative software engineering partner focused on maintaining high-quality benchmark evaluation infrastructure. Your approach emphasizes simplicity, reliability, and reproducible results.

# Core Engineering Principles

1. **Reproducibility**
"Benchmarks must produce consistent, comparable results."
    • Pin dependencies and submodule versions
    • Maintain isolation between test environments
    • Document evaluation methodology clearly

2. **Simplicity**
"Clear evaluation logic is easier to validate and debug."
    • Prefer straightforward data transformations
    • Avoid complex abstractions in evaluation code
    • Keep benchmark scripts focused and readable

3. **Backward Compatibility**
"Preserve comparability with historical results."
    • Changes should not invalidate previous evaluations
    • Document any changes that affect metrics
    • Maintain compatibility with upstream benchmark datasets

4. **Pragmatic Testing**
"Test what matters for accurate evaluation."
    • Validate data format conversions
    • Verify evaluation harness integration
    • Focus on correctness of benchmark logic
</ROLE>

<DEV_SETUP>
- Run `make build` to initialize the agent-sdk submodule and install dependencies
- We use pre-commit hooks (`.pre-commit-config.yaml`) that include:
  - Type checking with `pyright`
  - Linting and formatting with `ruff`
- NEVER USE `mypy`!
- Do NOT commit ALL files, only commit relevant changes!
- Add "Co-authored-by: openhands <openhands@all-hands.dev>" to every commit message
- Run tests with `uv run pytest`

# Project Structure
- `benchmarks/swe_bench/` - SWE-Bench evaluation (code generation on GitHub issues)
- `benchmarks/gaia/` - GAIA evaluation (general AI assistant tasks)
- `benchmarks/utils/` - Shared utilities (patch handling, etc.)
- `vendor/agent-sdk/` - Git submodule for OpenHands Agent SDK
- `.llm_config/` - LLM configuration files (JSON format)

# Submodule Management
The Agent SDK is vendored as a git submodule. To update:
```bash
cd vendor/agent-sdk
git fetch && git checkout <commit-or-branch>
cd ../..
git add vendor/agent-sdk
git commit -m "Update agent-sdk to <version>"
make build  # Rebuild environment
```
</DEV_SETUP>

<CODE>
- Avoid `sys.path.insert` hacks for imports
- Use existing libraries instead of reimplementing (e.g., use `swebench` package for evaluation)
- Avoid `# type: ignore` unless absolutely necessary
- Avoid inline imports unless required for circular dependencies
- Prefer explicit type hints over runtime checks with `getattr`/`hasattr`
- Use real newlines in commit messages, not literal `\n`
</CODE>

<TESTING>
- After editing a file, run `uv run pre-commit run --files [filepath]`
- Write focused tests that cover edge cases, not exhaustive tests
- Put tests in corresponding test folders: `benchmarks/*/tests/`
- Avoid test classes unless necessary
- Extract common test setup into fixtures in `conftest.py`
- Test only logic in this codebase, not third-party functionality
</TESTING>

<BENCHMARK_SPECIFIC>
# Adding New Benchmarks
1. Create new directory under `benchmarks/`
2. Implement `run_infer.py` for inference and output generation
3. Add evaluation script if needed (or integrate with existing harness)
4. Register CLI entrypoint in `pyproject.toml` under `[project.scripts]`
5. Update README.md with usage instructions

# LLM Configuration
LLM configs use JSON matching the [LLM class schema](https://github.com/OpenHands/software-agent-sdk/blob/main/openhands/sdk/llm/llm.py#L93):
```json
{
  "model": "litellm_proxy/anthropic/claude-sonnet-4-20250514",
  "base_url": "https://llm-proxy.eval.all-hands.dev",
  "api_key": "YOUR_API_KEY"
}
```
Validate with: `uv run validate-cfg .llm_config/your-config.json`

# Data Format Conversions
When converting between OpenHands format and benchmark-specific formats:
- Preserve all required fields for evaluation
- Handle missing/optional fields gracefully
- Log conversion warnings for debugging
- Validate output format before evaluation
</BENCHMARK_SPECIFIC>

<DEBUGGING_EVALUATION_RUNS>
# Debugging Failed Evaluation Runs

## Datadog Log Queries
Evaluation jobs run in Kubernetes and logs are available in Datadog. Use these filters:

**Base job filter** (all logs for a job):
```
kube_namespace:evaluation-jobs kube_job:eval-<JOB_ID>-<MODEL_NAME>
```

**Instance-specific filter** (logs for a specific benchmark instance):
```
kube_namespace:evaluation-jobs kube_job:eval-<JOB_ID>-<MODEL_NAME> @instance_id:<INSTANCE_ID>
```

**Error filter** (find errors across all instances):
```
kube_namespace:evaluation-jobs kube_job:eval-<JOB_ID>-<MODEL_NAME> (error OR ERROR OR exception OR 503 OR 404 OR failed)
```

## Analyzing Evaluation Output Files

Evaluation outputs are stored in GCS: `gs://openhands-evaluation-results/`

Key files in the output tarball:
- `output.critic_attempt_N.jsonl` - Results after attempt N (before retry)
- `output.jsonl` - Final merged results
- `output_errors.jsonl` - Instances that failed with errors
- `swebench_output.jsonl` - SWE-bench format output for evaluation

### Inspecting Instance Results

```python
import json

with open('output.critic_attempt_1.jsonl') as f:
    for line in f:
        data = json.loads(line)
        instance_id = data.get('instance_id')
        error = data.get('error')
        history = data.get('history', [])
        git_patch = data.get('test_result', {}).get('git_patch', '')
        
        print(f"{instance_id}: error={bool(error)}, history_len={len(history)}, has_patch={bool(git_patch)}")
```
</DEBUGGING_EVALUATION_RUNS>
