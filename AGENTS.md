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
- **Cleanroom workspace layout** (verified by inspecting agent runtime in retry-21):
    - `/workspace/.git/`, `/workspace/README.md`, etc. — cloned reference repo (sources only).
    - `/workspace/executable` — **the reference binary**, mode `---x--x--x` (execute-only,
      NOT readable). The `binary_path` rendered into `prompts/default.j2` (currently
      `/workspace/<repo_name>`) is **wrong**; the agent always finds the real binary at
      `/workspace/executable` via its own `ls`.
    - `/workspace/project/` — initially empty placeholder (legacy / unused).
    - The agent's working directory is `/workspace/`. `compile.sh` lives at
      `/workspace/compile.sh` and produces `/workspace/executable` — i.e. the agent's
      build literally **overwrites the reference binary** at `/workspace/executable`.
      By the time any Stop hook fires (end of conversation), the reference is gone.
- **Reference-diffs hook gotcha** (retry-21 lesson): a Stop hook that diffs
  `$REF --help` against `./executable --help` cannot work if it tries to use
  `/workspace/executable` as `$REF` — because the agent's compile.sh has replaced
  it. Two paths forward:
    1. Capture `executable --help` / `executable -h` into a hidden, read-only
       location (e.g. `/opt/programbench-ref/`) **before the conversation starts**
       (e.g. via a pre-conversation `WorkspaceClient.bash` call in `run_infer.py`),
       then have the Stop hook diff against those captured outputs.
    2. Tell the agent in the prompt to `mv /workspace/executable
       /workspace/executable.ref` before building (some agents already do this
       spontaneously; we observed it in zoxide retry-21).
  Approach (1) is robust to agent behaviour; approach (2) keeps the hook simple
  but depends on agent compliance. **Retry-22 shipped approach (2)** with a
  Step-0 prominent block at the top of `prompts/default.j2`; Sonnet 4.5
  complied 3-for-3 on the smoke set.

- **Reference-diffs hook v2** (retry-22 -> retry-23): the v1 hook only diffed
  top-level `--help` and `-h`. Bucketing R22's residual 352 failures showed
  68% are reachable by expanding the probe set. v2 adds:
    1. **Top-level invalid flag probe** (`<bin> --__bogus__`) — catches argv
       parser leaks (agent silently accepts unknown flags rc=0 where ref rc=2).
    2. **Subcommand discovery** via awk parsing of the reference's
       `Commands:` / `Subcommands:` / `Available Commands:` /
       `Available subcommands:` section. Capped at
       `PB_REFERENCE_DIFFS_MAX_SUBCMDS` (default 8).
    3. **Per-subcommand probes**: `<sub> --help` (drift detection),
       `<sub> --__bogus__` (validation gap), `<sub> /<bogus-path>`
       (validation gap). Compares both rc and stderr/stdout.
    4. **argv[0] normalization** via `bash -c 'exec -a "$1" "${@:2}"' _
       executable "$bin" "$@"`. Both ref and agent see argv[0]="executable",
       so binaries that derive `Usage:` from argv[0] (clap default) don't
       false-positive on basename drift. **Note:** this only works for ELF
       binaries — shell scripts get $0 from the kernel exec path, not from
       `exec -a`. ProgramBench reference binaries are always compiled, so
       we're safe in production.
  Hook timeout was bumped 120s -> 240s to fit the worst-case probe count
  (3 top-level @ 30s + 8 subs * 3 probes @ 5s = ~185s).
  Smoke-tested with synthetic gcc-built C binaries; see
  `tests/test_programbench.py::TestReferenceDiffsHookV2`.

# SWE-Bench Multimodal Notes
- The default `swebenchmultimodal-infer` selection now comes from `benchmarks/swebenchmultimodal/resolved_instances.txt`.
- `resolved_instances.txt` is generated from `ambiguity_annotations.json` and contains all instances annotated with the `SOLVEABLE` keyword.
- `benchmarks/swebenchmultimodal/build_images.py` does not inherit that default automatically; pass `--select benchmarks/swebenchmultimodal/resolved_instances.txt` when you need matching image builds.

# SWE-Bench Pro Notes
- `ScaleAI/SWE-bench_Pro` exposes the official base image tag in each row's `dockerhub_tag` field; build and inference code should derive base images from that field instead of `swebench.harness.constants.MAP_VERSION_TO_INSTALL`.
- SWE-Bench Pro agent images expose the checked-out repository at `/app`, not `/testbed`, so inference must copy from `/app` into the workspace before resetting to `base_commit`.
- The official harness lives at `scaleapi/SWE-bench_Pro-os`; the repo-local wrapper converts OpenHands `output.jsonl` to the upstream patch JSON format and then invokes `swe_bench_pro_eval.py`.
- Upstream's `eval_with_docker` bind-mounts each instance's `workspace_dir` (via `os.path.abspath`) into the per-instance test container — unlike `swebench`/`swtbench` which use `put_archive`/`get_archive` (tar in/out). Under a DinD sidecar (separate filesystem from the eval container), that bind source resolves to nothing on the dockerd side, so the container starts with an empty `/workspace`, can't find `entryscript.sh`, and emits zero output. Fix: put the harness's input + workspace under a volume that's mounted at the same path in both containers (we use the `dind-shared` emptyDir at `/shared`).
- Laminar's `update_evaluation_scores` makes one API call per instance after the harness finishes; it can silently kill the wrapper interpreter on multi-instance runs (no traceback in the log) — wrap the call in `except BaseException` and keep a bash-side fallback that uses the on-disk report file if the wrapper exits non-zero but the report exists. Telemetry must never sink a valid evaluation.


</BENCHMARK_SPECIFIC>
