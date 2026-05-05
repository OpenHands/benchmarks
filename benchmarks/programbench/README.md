# ProgramBench

[ProgramBench](https://programbench.com) (Yang et al., 2026) asks: *can a
language-model agent rebuild a program from scratch given only the compiled
binary and its public documentation?* The benchmark ships 200 cleanroom
tasks (and an extended set), each as a Docker image containing the binary
plus its docs.

This module wraps the upstream
[facebookresearch/ProgramBench](https://github.com/facebookresearch/ProgramBench)
harness so the OpenHands [Software Agent SDK](https://github.com/OpenHands/software-agent-sdk)
can be used as the inference agent.

## How it works

1. **Inference (`programbench-infer`)** loads the upstream task list, layers
   `openhands-agent-server` on top of each `programbench/<id>:task_cleanroom`
   image, and runs the SDK agent with no internet access. After the agent
   finishes, `/workspace` is tarred up into
   `<eval_output_dir>/run/<instance_id>/submission.tar.gz` — exactly the
   layout `programbench eval` expects.
2. **Evaluation (`programbench-eval`)** shells out to the upstream
   `programbench eval <run_dir>` CLI, then aggregates the per-instance
   `<id>/<id>.eval.json` files into our standard report format (`resolved`,
   `almost_resolved`, `error`, …).

## Prerequisites

- Linux x86_64 host. The upstream task images are built for `linux/amd64`
  only and emulating them via QEMU is impractically slow.
- Docker daemon running and reachable to the user invoking the script.
- The `programbench` Python package (added as a dependency in this repo's
  `pyproject.toml`).
- An LLM config under `.llm_config/`.

## Usage

### Inference

```bash
# Smoke test — first 5 tasks
uv run programbench-infer .llm_config/claude.json --n-limit 5

# Selected subset of tasks
uv run programbench-infer .llm_config/claude.json \
    --select benchmarks/programbench/instances.txt

# Higher concurrency, more iterations
uv run programbench-infer .llm_config/claude.json \
    --n-limit 20 --num-workers 4 --max-iterations 300
```

### Evaluation

```bash
uv run programbench-eval ./eval_outputs/.../output.jsonl
```

Pass `--skip-eval` to re-aggregate an already-graded run without rerunning
the upstream harness, and `--force` to regrade everything.

## Output layout

```
eval_outputs/
└── programbench__ProgramBench-test/
    └── <model>_sdk_<sha>_maxiter_200/
        ├── metadata.json
        ├── output.jsonl
        ├── output.report.json
        └── run/
            ├── abishekvashok__cmatrix.5c082c6/
            │   ├── submission.tar.gz
            │   └── abishekvashok__cmatrix.5c082c6.eval.json
            └── …
```

## Caveats

- **Offline inference.** ProgramBench requires the agent to have no
  internet access. We enforce `network=none` on the agent's Docker
  container by default; pass `--allow-network` only when debugging — it
  invalidates leaderboard comparability.
- **Image pulls are large.** Each task image is multiple GiB. Plan disk
  budget accordingly.
- **Remote workspace** is not yet wired up for ProgramBench because we
  have no reliable network-isolation hook for the runtime API. PRs welcome.

## References

- ProgramBench paper & leaderboard: <https://programbench.com>
- Upstream harness: <https://github.com/facebookresearch/ProgramBench>
- Upstream usage guide: <https://github.com/facebookresearch/ProgramBench/blob/main/docs/README.md>
