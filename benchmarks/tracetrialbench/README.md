# TraceTrial-Bench

[TraceTrial-Bench](https://github.com/TraceLite-AI/TraceTrial-Bench) ([arXiv:2610.32285](https://arxiv.org/abs/2610.32285), [Dataset](https://huggingface.co/datasets/TraceLite-AI/TraceTrial-Bench)) evaluates **AI agent harness reliability and failure attribution** across **95 tasks**, **40 mechanism nodes**, and **7 reliability dimensions**:

- **C — Context:** Truncation accounting, entity retention under condensation, cross-turn persistence
- **E — Execution:** Exit-code fidelity, hard/idle timeout semantics, process containment, interactive PTY sessions
- **G — Governance:** Approval mediation, fail-closed sub-agent confirmation, secret redaction across rotation
- **L — Lifecycle:** Multi-agent delegation, background task completion, session recovery across restarts
- **O — Observability:** Sub-agent event forwarding, hook audit trails, structured error channels
- **T — Tools:** MCP tool discovery, reserved-name collision isolation, `EmbeddedResource` & `structuredContent` fidelity
- **V — Verification:** Completion claims versus authoritative workspace state

Unlike single-layer pass/fail benchmarks, TraceTrial-Bench scores **Agent** task completion (50 pts) and **Harness** mechanism reliability (50 pts) independently from execution traces and attributes failures to `clean_pass`, `harness_caused`, `agent_caused`, `co_caused`, or `harness_defect_absorbed`.

## Prerequisites

1. Build the OpenHands benchmarks workspace (initializes `vendor/software-agent-sdk`):
   ```bash
   make build
   ```
2. Docker daemon available on the host (each task runs inside an isolated container with task-scoped Unix-socket simulators, MCP servers, and trace recorders).

By default (`--use-vendor-sdk`), `tracetrialbench-infer` overlays the checked-out `vendor/software-agent-sdk` (`openhands-sdk` and `openhands-tools`) into the evaluation container image so CI and local runs measure the exact SDK revision under test. Pass `--no-use-vendor-sdk` to evaluate the published PyPI `openhands` package instead.

## Usage

### 1. Run a 5-task smoke subset

```bash
uv run tracetrialbench-infer .llm_config/eval.json --n-limit 5
```

### 2. Run specific mechanism tasks (`--select`)

```bash
printf "C1-escalation-export-001\nT3-reserved-tool-name-001\n" > selected_tasks.txt
uv run tracetrialbench-infer .llm_config/eval.json --select selected_tasks.txt
```

### 3. Run the full 95-task paper benchmark

```bash
uv run tracetrialbench-infer .llm_config/eval.json --num-workers 4
```

### 4. Generate the evaluation report

```bash
uv run tracetrialbench-eval eval_outputs/<run_dir>/output.jsonl
```

To require both Agent and Harness criteria (`Clean` pass) for `resolved_ids`:

```bash
uv run tracetrialbench-eval eval_outputs/<run_dir>/output.jsonl --criterion clean_pass
```

## Output Format

- `output.jsonl`: One `EvalOutput`-compatible JSON record per task containing token usage, cost, and `test_result` (`agent_pass`, `harness_pass`, `clean_pass`, `agent_score_pct`, `harness_score_pct`, `overall_score`, `attribution`, `dimension`).
- `output.report.json`: Standard OpenHands summary (`total_instances`, `completed_instances`, `resolved_instances`, `aggregate_metrics`) plus `tracetrial_metrics` (`agent_pass_rate`, `clean_pass_rate`, `agent_score_pct`, `harness_macro_pct`, `overall_score_pct`, `per_dimension`, `attribution_counts`).

## Citation

```bibtex
@misc{tracelite2026trialbytrace,
  title         = {Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution},
  author        = {Cao, Zhiyong and Zhu, Hengjie and Wang, Yongxin and others},
  year          = {2026},
  eprint        = {2610.32285},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.32285}
}
```
