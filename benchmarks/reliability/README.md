# Fault-injection reliability benchmark

This package implements the benchmark core proposed in
[RFC #764](https://github.com/OpenHands/benchmarks/issues/764).

It runs a matched no-fault baseline and faulted run, places faults from a
seeded declarative schedule after persisted SDK events, records external-effect
evidence independently of the agent, and emits inspectable JSON and Markdown
scorecards.

## What is implemented

- strict, versioned JSON scenario loading
- deterministic event-trigger matching with duplicate-event suppression
- explicit adapters for sandbox restart, lost dispatch response, `SIGKILL`
  mid-tool-call, and network partition/release
- durable event, fault, recovery, effect, task, and metric artifacts
- a scenario-owned append-only effect ledger with canonical payload digests
- deterministic completion/resume, duplicate-effect, and recovery-overhead
  graders
- JSON and Markdown scorecards with raw component results
- an OpenHands SDK callback adapter that preserves persist-before-inject order
- a CLI entrypoint and end-to-end tests covering all four fault categories

The core does not pretend that pausing a process is a sandbox restart or that
raising an exception before dispatch is a lost response. A benchmark-specific
adapter must implement the real infrastructure operation at a named, verified
boundary and return details for the fault receipt.

## Scenario

See [`scenarios/example.json`](scenarios/example.json). Every trigger names a
persisted event type and 1-based ordinal. An optional `tool_call_id` narrows the
match. Fault IDs must be unique.

The seed deterministically orders faults that share a trigger. Replaying the
same event ID never fires a fault twice.

## Adapter contract

An adapter implements `ReliabilityAdapter.open_run` and returns a `RunSession`.
The session:

1. executes one independent task run;
2. publishes persisted events synchronously through `on_event`;
3. implements the four explicit `FaultContext` operations;
4. writes intended and committed external operations through
   `EffectLedger(artifacts.effects_path)`;
5. returns native task success, recovery receipts, and raw metrics.

Use `build_reliability_event_callback` to compose the reliability callback with
the existing benchmark event-persistence callback. Existing callbacks run
first, so a fault cannot fire before its trigger event is durable.

For a recovery to score as resume, the adapter must emit successful
`conversation_history_restored` and `runtime_reattached` receipts. A run that
sets `fresh_retry_used=True` fails the resume grader even if the ordinary task
eventually passes.

## Run

Register a zero-argument adapter factory and run:

```bash
uv run reliability-eval \
  --scenario benchmarks/reliability/scenarios/example.json \
  --adapter your_package.reliability_adapter:create_adapter \
  --output-dir evaluation_outputs/reliability
```

The adapter is loaded with ordinary `importlib`; no `sys.path` mutation is
performed.

## Artifacts

Each baseline and faulted run contains:

- `manifest.json`
- `events.jsonl`
- `fault_receipts.jsonl`
- `recovery_receipts.jsonl`
- `effects.jsonl`
- `task_result.json`
- `metrics.json`
- `reliability_result.json` for the faulted run

The output root also receives `scorecard.json` and `scorecard.md`.

## Grading

Completion/resume requires:

- the ordinary benchmark task to pass;
- every scheduled fault to have an `applied` receipt;
- no injection failure;
- history-restored and runtime-reattached recovery receipts; and
- no forbidden fresh retry.

The duplicate-effect grader compares effect `intent` and `commit` records by
logical operation ID and canonical payload digest. It reports missing,
unexpected, mismatched, and excess commits separately.

Recovery overhead compares the faulted run against its matched baseline and
reports wall-time delta/ratio, recovery time, and iteration, event, tool-call,
token, and cost deltas. Missing or zero baselines remain explicit rather than
being coerced.

## Validate

```bash
uv run pre-commit run --files benchmarks/reliability tests/test_reliability.py
uv run pytest tests/test_reliability.py
```

## Context

- [Architecture](ARCHITECTURE.md)
- [Design](DESIGN.md)
- [`OpenHands/benchmarks#488`](https://github.com/OpenHands/benchmarks/issues/488)
- [`OpenHands/OpenHands#14260`](https://github.com/OpenHands/OpenHands/issues/14260)
- [`OpenHands/OpenHands#13578`](https://github.com/OpenHands/OpenHands/issues/13578)
