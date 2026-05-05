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
- See [CONTRIBUTING.md](./CONTRIBUTING.md) for benchmark conventions and contribution guidelines

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

# Terminal-Bench Notes
- Harbor's installable package is `harbor` (not `harbor-bench`).
- The Harbor dataset name used in CI is `terminal-bench@2.0`.
- For CI smoke tests, pass `--n-limit <count>` to `terminalbench-infer` so Harbor only runs the requested subset.

# ProgramBench Notes
- Upstream package is `programbench` (PyPI). Pinned `>=1.0,<2.0` in `pyproject.toml` (skipped on macOS — upstream images are linux/amd64 only).
- Task images live at `programbench/<owner>_1776_<repo>.<sha>:<tag>` on Docker Hub. The agent runs against `:task_cleanroom`; evaluation runs against `:task`.
- The `__` separator in instance ids is replaced with `_1776_` for Docker tag compatibility (see `_instance_to_image`).
- **Strict offline isolation is not yet enforced** (known limitation). `--network=none` breaks the SDK's HTTP control channel and `docker network create --internal` breaks `-p` port mapping; the proper fix is in-container egress filtering with `CAP_NET_ADMIN` + iptables in an init step. Until that lands, the agent container uses the default Docker bridge and we rely on the system prompt + cleanroom image to keep the agent honest. `--allow-network` is reserved so future strict-offline runs are distinguishable in metadata. Treat current results as engineering-grade, not leaderboard-faithful.
- `programbench-infer` writes submission tarballs to `<eval_output_dir>/run/<instance_id>/submission.tar.gz`; this matches the layout the upstream `programbench eval` CLI consumes.
- The 200-task base set is loaded via `programbench.utils.load_data.load_all_instances(include_tests=False)`. Use `include_tests=False` during inference because the tests blob is large and only needed by the eval harness.
- CI smoke runs the first 5 instances (matches `benchmarks/programbench/instances.txt`).

# SWE-Bench Multimodal Notes
- The default `swebenchmultimodal-infer` selection now comes from `benchmarks/swebenchmultimodal/resolved_instances.txt`.
- `resolved_instances.txt` is generated from `ambiguity_annotations.json` and contains all instances annotated with the `SOLVEABLE` keyword.
- `benchmarks/swebenchmultimodal/build_images.py` does not inherit that default automatically; pass `--select benchmarks/swebenchmultimodal/resolved_instances.txt` when you need matching image builds.

</BENCHMARK_SPECIFIC>
