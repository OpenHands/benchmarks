# Fault-injection reliability benchmark architecture

Status: verified architecture for the implemented benchmark core and its
infrastructure adapter boundary.

## Evidence base

The architecture was checked against:

- `OpenHands/benchmarks` at `4e5469e0caaf54d1ad827d18b524bdfb79d58430`
- `OpenHands/software-agent-sdk` at
  `6d597ff7d5d3c89ef8ba0c8e3b3c6a09169da07c`
- `OpenHands/benchmarks#488`
- `OpenHands/OpenHands#14260` and `OpenHands/OpenHands#13578`

The source was first inspected through the connected GitHub app at the pinned
revisions, then checked against local clones before this implementation was published.

## Existing execution path

For SWE-bench, the verified path is:

1. `benchmarks/utils/evaluation.py:Evaluation.run`
2. `Evaluation._run_iterative_mode_async`
3. `Evaluation._run_attempt_async`
4. `Evaluation._process_one_sync`
5. `Evaluation._execute_single_attempt`
6. `benchmarks/swebench/run_infer.py:SWEBenchEvaluation.prepare_workspace`
7. `SWEBenchEvaluation.evaluate_instance`
8. `benchmarks/utils/fake_user_response.py:
   run_conversation_with_fake_user_response`
9. SDK `Conversation.run` / `RemoteConversation.run`
10. agent response dispatch, persisted action event, tool execution, and
    persisted observation/error event

`Evaluation._execute_single_attempt` is the runner-level lifecycle boundary. It
creates the workspace, calls the benchmark-specific evaluator, captures failure
artifacts, classifies retries, and cleans up. It is appropriate for scenario
ownership, coarse runtime faults, and fault receipts. It is not precise enough
to model an ambiguous tool outcome.

`SWEBenchEvaluation.prepare_workspace` is the concrete workspace selection seam.
It constructs `DockerWorkspace`, `ApptainerWorkspace`, or
`APIRemoteWorkspace`, using an eval agent-server image. This is where a
scenario-scoped lifecycle adapter can be attached without changing task
grading.

`SWEBenchEvaluation.evaluate_instance` creates the SDK `Conversation` with
`build_event_persistence_callback(...)`, sends the instruction, and enters
`run_conversation_with_fake_user_response`. The callback and returned event
history are the primary inspectable trigger/evidence stream.

## Verified SDK state and dispatch model

`openhands-sdk/openhands/sdk/conversation/state.py:
ConversationState.append_event` is the event storage chokepoint. It appends to
`EventLog` and advances the active event leaf.

`conversation/event_store.py:EventLog.append` writes one serialized event under
a file-store lock and rejects duplicate event IDs. `ConversationState.create`
opens or creates the store, reloads `base_state.json`, attaches the event log,
and rebuilds derived state.

The relevant ambiguous-outcome window is verified in:

- `openhands-sdk/openhands/sdk/agent/response_dispatch.py:
  _handle_tool_calls` / `_ahandle_tool_calls`
- `openhands-sdk/openhands/sdk/agent/agent.py:_ActionBatch.prepare`
- `agent.py:_ActionBatch.emit`
- `agent.py:_execute_action_event`

The dispatcher emits an `ActionEvent` through `on_event` before executing the
tool. Observations or errors are emitted only after the tool runner returns.
Therefore a process can die after an irreversible external effect commits but
before its observation is persisted.

`Agent.step` checks `ConversationState.get_unmatched_actions` before sampling a
new response and can execute unmatched actions. This is a useful replay seam
and also the source of duplicate-effect risk in ambiguous outcomes.

Agent-server cold load behaves differently. In
`openhands-agent-server/openhands/agent_server/event_service.py:
EventService.start`, a persisted conversation left `RUNNING` is changed to
`ERROR`; the server emits an `AgentErrorEvent` for the first unmatched action.
That prevents blind replay of that action on this path, but does not establish
whether the external effect committed. Parallel batches may leave additional
unmatched actions, so the benchmark must record actual behavior rather than
assume all orphans are reconciled.

## Fault hook map

### Sandbox restart

Runner ownership:

- `Evaluation._execute_single_attempt`
- `SWEBenchEvaluation.prepare_workspace`

Local runtime seams:

- `openhands-workspace/openhands/workspace/docker/workspace.py:
  DockerWorkspace._start_container`
- `DockerWorkspace.cleanup`, `pause`, and `resume`

Remote runtime seams:

- `openhands-workspace/openhands/workspace/remote_api/workspace.py:
  _start_or_attach_to_runtime`
- `_start_runtime`, `_resume_runtime`, `pause`, and `resume`

A restart injector must stop/replace the agent-server container or remote
runtime and then exercise the real attach/load path. Docker pause/unpause is
not a restart and must be reported as a separate fault if used.

### Lost dispatch response

The benchmark wrapper around `Conversation.run` can drop a run-level result,
but that does not prove a tool dispatch committed. The precise in-process seam
is after `_execute_action_event` returns and before `_ActionBatch.emit`
persists the resulting observation.

For remote runs, the production request/response seam still needs confirmation
in the remote conversation or workspace client before implementation. The
`APIRemoteWorkspace._send_api_request` seam controls runtime lifecycle calls,
not every tool call to the agent server. The scenario must state whether it
models:

- a lost tool result before observation persistence, or
- a lost client/agent-server transport response.

They are not interchangeable.

### SIGKILL mid-tool-call

Trigger on the persisted `ActionEvent`, then kill the process/container while
`_execute_action_event` is active and before an observation/error exists.

`LocalConversation.interrupt()` is not a substitute for SIGKILL. It sets a
cancellation token and cancels the tracked async task; worker threads may
continue until their tools return. A real SIGKILL bypasses cleanup and must be
implemented at the process/container layer.

### Network partition

Runtime-control partitions can wrap
`APIRemoteWorkspace._send_api_request`. Agent-server tool/run traffic requires
the corresponding remote client seam to be verified before implementation.
Partitions need an explicit direction, endpoint class, start trigger, and
duration or release trigger.

`benchmarks/utils/acp.py:workspace_keepalive` is an existing observation point.
It executes `true` every 60 seconds and suppresses failures. Issue #488 shows it
is not a sufficient recovery mechanism.

## Outcome judgment

Task correctness stays with the existing benchmark grader. For SWE-bench,
`benchmarks/swebench/eval_infer.py:run_swebench_evaluation` invokes
`python -m swebench.harness.run_evaluation`. SDK critics only gate retries or
rank candidates; they are not the official correctness oracle.

Reliability grading consumes inspectable artifacts:

- terminal `ConversationExecutionStatus`
- ordinary task-grader result
- persisted event log and callback event stream
- scheduled and observed fault receipts
- runtime/conversation identity before and after recovery
- scenario-owned external effect ledger
- baseline and faulted timing, iteration, event, LLM, and tool-call metrics

Completion/resume passes only when the ordinary task passes and the run shows a
real reattachment/recovery path. Starting the instance over in a fresh
workspace is a retry, not resume.

No-duplicate-effect grading cannot be inferred from the SDK event log alone.
Each irreversible-action scenario must expose an inspectable ledger or
query-by-idempotency-key oracle independent of the agent. The grader compares
committed effects against intended logical operations.

Recovery overhead compares a faulted run with a matched no-fault baseline for
the same scenario, seed, instance, agent, model, and resource configuration.
Raw measures include wall time, recovery-window time, iterations, event count,
tool calls, token/cost metrics, and replay/detection time.

## Persistence and restart receipts

Issue #14260 demonstrates that durable OpenHands events do not imply provider
session recovery. Acceptance evidence should distinguish:

- `conversation_history_restored`
- `provider_session_rebound`

The second requires the same provider session ID/cwd and preserved provider
storage, not merely a bootstrap prompt containing prior messages.

Issue #13578 demonstrates that successful in-memory agent activity is not
evidence of durable recovery. The event store must be visible after app/server
restart. That issue narrows the startup timeout only to app-server/agent-server
connectivity; it does not identify a more specific root cause, so the benchmark
must not encode one as fact.

## Existing recovery measurement to reuse

`scripts/event_sourcing_benchmarks/bench_replay_and_recovery.py` already
measures event deserialization, replay, and unmatched-action scan cost on real
SWE-bench traces. Reuse its concepts and data where possible. It does not inject
live faults, judge end-to-end task completion, or detect duplicate external
effects.

## Placement

The first milestone should live as a benchmark package under
`benchmarks/reliability`. Scenario adapters remain package-local until at least
one other benchmark needs them. Only then should generic components move to
`benchmarks/utils`, consistent with the repository's contribution guidance.

SDK changes should be limited to narrow, reusable injection hooks or observable
receipts that cannot be implemented at the benchmark layer. The current
`build_reliability_event_callback` composes with SDK callbacks without changing
the vendored SDK.

## Adapter-specific follow-up

- The exact remote client seam for lost tool responses and agent-server network
  partitions.
- Whether cold-start recovery should reconcile all unmatched parallel actions
  or intentionally fail closed after the first.
- Which irreversible test action is acceptable for the initial milestone.
- Whether provider-session rebound is in scope for V1 or reported separately.
