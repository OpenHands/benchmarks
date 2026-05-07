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

# Selected subset of tasks: pass a newline-separated instance-id file
uv run programbench-infer .llm_config/claude.json \
    --select my_instances.txt

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
    └── <model>_sdk_<sha>_maxiter_1000/
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

- **Offline inference (known limitation).** ProgramBench's leaderboard
  rules require the agent to have no internet access during inference.
  Enforcing that via Docker is harder than it sounds: `--network none`
  breaks the SDK's HTTP control channel (Docker port mapping needs a
  network interface), and `docker network create --internal` blocks
  the `-p` mapping too. We currently rely on the system prompt + the
  cleanroom image (which ships everything the task needs) and leave
  the container on the default Docker bridge. Strict in-container
  egress filtering (iptables in an init step with `CAP_NET_ADMIN`) is
  tracked as follow-up work in `AGENTS.md`. The `--allow-network` flag
  is reserved so that future strict-offline runs are distinguishable in
  metadata. **Until that lands, treat results as engineering-grade, not
  leaderboard-faithful.**
- **Image pulls are large.** Each task image is multiple GiB. Plan disk
  budget accordingly.
- **Remote workspace** is not yet wired up for ProgramBench because we
  have no reliable network-isolation hook for the runtime API. PRs welcome.

## References

- ProgramBench paper & leaderboard: <https://programbench.com>
- Upstream harness: <https://github.com/facebookresearch/ProgramBench>
- Upstream usage guide: <https://github.com/facebookresearch/ProgramBench/blob/main/docs/README.md>
