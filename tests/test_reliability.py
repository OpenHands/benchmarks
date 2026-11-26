"""Focused tests for the deterministic reliability benchmark core."""

from collections.abc import Callable
from pathlib import Path

import pytest

from benchmarks.reliability.artifacts import RunArtifacts, read_json, read_jsonl
from benchmarks.reliability.effects import EffectLedger, payload_digest
from benchmarks.reliability.injectors import FaultInjectionError
from benchmarks.reliability.models import (
    EventRecord,
    EventTrigger,
    FaultKind,
    FaultSpec,
    JsonObject,
    RecoveryReceipt,
    RunIdentity,
    RunMetrics,
    RunObservation,
    Scenario,
)
from benchmarks.reliability.reporting import (
    write_scorecard_json,
    write_scorecard_markdown,
)
from benchmarks.reliability.runner import ReliabilityAdapter, RunSession, run_scenario
from benchmarks.reliability.scenario import load_scenario
from benchmarks.reliability.schedule import compile_schedule
from benchmarks.reliability.sdk_events import build_reliability_event_callback
from openhands.sdk.event import Event, PauseEvent


def test_load_json_scenario_and_reject_unknown_fields(tmp_path: Path) -> None:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        """
{
  "schema_version": 1,
  "scenario_id": "restart-on-action",
  "task": {
    "benchmark": "swebench",
    "instance_id": "project__issue-1"
  },
  "agent": {"config": "default"},
  "seed": 17,
  "faults": [
    {
      "fault_id": "restart",
      "kind": "sandbox_restart",
      "trigger": {
        "event": {"event_type": "ActionEvent", "ordinal": 1}
      },
      "parameters": {"mode": "hard"}
    }
  ]
}
""".lstrip(),
        encoding="utf-8",
    )

    scenario = load_scenario(scenario_path)

    assert scenario.scenario_id == "restart-on-action"
    assert scenario.faults[0].kind == FaultKind.SANDBOX_RESTART
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        scenario_path.read_text(encoding="utf-8").replace(
            '"schema_version": 1,',
            '"schema_version": 1, "unknown": true,',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields: unknown"):
        load_scenario(invalid_path)


def test_schedule_is_seeded_deterministic_and_ignores_duplicate_events() -> None:
    scenario = _scenario()
    first = compile_schedule(scenario)
    second = compile_schedule(scenario)
    event = _event("action-1", ordinal=1)

    first_match = first.observe((event,))
    second_match = second.observe((event,))

    assert [fault.fault_id for fault in first_match] == [
        fault.fault_id for fault in second_match
    ]
    assert set(fault.fault_id for fault in first_match) == {
        "restart",
        "lost-response",
        "sigkill",
        "partition",
    }
    assert first.observe((event,)) == ()
    assert first.pending() == ()


def test_effect_ledger_uses_canonical_payload_digests(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path / "effects.jsonl")

    intent = ledger.record_intent(
        operation_id="publish-1",
        payload={"b": 2, "a": 1},
        monotonic_seconds=1.0,
    )
    commit = ledger.record_commit(
        operation_id="publish-1",
        payload={"a": 1, "b": 2},
        monotonic_seconds=2.0,
    )

    assert intent.payload_digest == commit.payload_digest
    assert intent.payload_digest == payload_digest({"a": 1, "b": 2})
    assert len(ledger.records()) == 2


def test_sdk_callback_persists_before_exposing_trigger() -> None:
    order: list[str] = []
    records: list[EventRecord] = []

    def persist(event: Event) -> None:
        order.append(f"persist:{event.id}")

    def observe(record: EventRecord) -> None:
        order.append("reliability")
        records.append(record)

    callback = build_reliability_event_callback(
        observe,
        persistence_callbacks=(persist,),
        monotonic=lambda: 12.5,
    )
    event = PauseEvent(id="pause-1")

    callback(event)

    assert order == ["persist:pause-1", "reliability"]
    assert records == [
        EventRecord(
            event_id="pause-1",
            event_type="PauseEvent",
            monotonic_seconds=12.5,
            details={
                "source": "user",
                "timestamp": event.timestamp,
            },
        )
    ]


def test_end_to_end_run_injects_all_faults_and_scores_artifacts(
    tmp_path: Path,
) -> None:
    adapter = _FakeAdapter(duplicate_effect=False)

    result = run_scenario(
        _scenario(),
        adapter=adapter,
        output_root=tmp_path,
    )

    assert result.task_passed
    assert result.completion_resume.passed
    assert result.no_duplicate_effect.passed
    assert result.recovery_overhead.passed
    fault_root = result.run.artifact_dir
    receipts = read_jsonl(fault_root / "fault_receipts.jsonl")
    assert sum(item["status"] == "applied" for item in receipts) == 4
    assert sum(item["status"] == "released" for item in receipts) == 1
    assert adapter.faulted_session is not None
    assert set(adapter.faulted_session.calls[:-1]) == {
        "restart_sandbox",
        "drop_dispatch_response",
        "sigkill_mid_tool_call",
        "partition_network",
    }
    assert adapter.faulted_session.calls[-1] == "heal_network"
    overhead = result.recovery_overhead.value
    assert isinstance(overhead, dict)
    assert overhead["wall_time_ratio"] == pytest.approx(1.5)
    assert read_json(fault_root / "reliability_result.json")["task_passed"] is True


def test_end_to_end_duplicate_effect_is_visible_not_inferred(
    tmp_path: Path,
) -> None:
    result = run_scenario(
        _scenario(),
        adapter=_FakeAdapter(duplicate_effect=True),
        output_root=tmp_path,
    )

    assert not result.no_duplicate_effect.passed
    assert result.no_duplicate_effect.reason_codes == ("duplicate_effect",)
    value = result.no_duplicate_effect.value
    assert isinstance(value, dict)
    assert value["excess_commits"] == 1


def test_scorecards_include_raw_components(
    tmp_path: Path,
) -> None:
    result = run_scenario(
        _scenario(),
        adapter=_FakeAdapter(duplicate_effect=False),
        output_root=tmp_path / "runs",
    )
    json_path = tmp_path / "scorecard.json"
    markdown_path = tmp_path / "scorecard.md"

    write_scorecard_json((result,), json_path)
    write_scorecard_markdown((result,), markdown_path)

    scorecard = read_json(json_path)
    summary = scorecard["summary"]
    assert isinstance(summary, dict)
    assert summary["completion_resume_passes"] == 1
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Completion/resume passes: 1" in markdown
    assert result.run.run_id in markdown


def test_missing_recovery_receipts_cannot_score_as_resume(tmp_path: Path) -> None:
    result = run_scenario(
        _scenario(),
        adapter=_FakeAdapter(
            duplicate_effect=False,
            emit_recovery=False,
        ),
        output_root=tmp_path,
    )

    assert result.task_passed
    assert not result.completion_resume.passed
    assert "recovery_receipt_missing" in result.completion_resume.reason_codes


def test_failed_injection_is_recorded_before_run_fails(tmp_path: Path) -> None:
    with pytest.raises(FaultInjectionError, match="failed to inject restart"):
        run_scenario(
            _scenario(),
            adapter=_FakeAdapter(
                duplicate_effect=False,
                fail_restart=True,
            ),
            output_root=tmp_path,
        )

    receipts = read_jsonl(
        tmp_path / "e2e-faults" / "23" / "faulted" / "fault_receipts.jsonl"
    )
    restart_receipts = [
        receipt for receipt in receipts if receipt["fault_id"] == "restart"
    ]
    assert restart_receipts[0]["status"] == "failed"
    assert "RuntimeError: restart failed" in str(restart_receipts[0]["details"])


def test_missing_effect_evidence_cannot_score_as_exactly_once(
    tmp_path: Path,
) -> None:
    result = run_scenario(
        _scenario(),
        adapter=_FakeAdapter(
            duplicate_effect=False,
            emit_effects=False,
        ),
        output_root=tmp_path,
    )

    assert not result.no_duplicate_effect.passed
    assert result.no_duplicate_effect.reason_codes == ("effect_evidence_missing",)


class _FakeAdapter(ReliabilityAdapter):
    def __init__(
        self,
        *,
        duplicate_effect: bool,
        emit_recovery: bool = True,
        fail_restart: bool = False,
        emit_effects: bool = True,
    ) -> None:
        self.duplicate_effect = duplicate_effect
        self.emit_recovery = emit_recovery
        self.fail_restart = fail_restart
        self.emit_effects = emit_effects
        self.faulted_session: _FakeSession | None = None

    def open_run(
        self,
        *,
        scenario: Scenario,
        run: RunIdentity,
        artifacts: RunArtifacts,
        baseline: bool,
    ) -> RunSession:
        session = _FakeSession(
            run=run,
            artifacts=artifacts,
            baseline=baseline,
            duplicate_effect=self.duplicate_effect,
            emit_recovery=self.emit_recovery,
            fail_restart=self.fail_restart,
            emit_effects=self.emit_effects,
        )
        if not baseline:
            self.faulted_session = session
        return session


class _FakeSession(RunSession):
    def __init__(
        self,
        *,
        run: RunIdentity,
        artifacts: RunArtifacts,
        baseline: bool,
        duplicate_effect: bool,
        emit_recovery: bool,
        fail_restart: bool,
        emit_effects: bool,
    ) -> None:
        self._run = run
        self.artifacts = artifacts
        self.baseline = baseline
        self.duplicate_effect = duplicate_effect
        self.emit_recovery = emit_recovery
        self.fail_restart = fail_restart
        self.emit_effects = emit_effects
        self.clock = 0.0
        self.calls: list[str] = []
        self.ledger = EffectLedger(artifacts.effects_path)

    @property
    def run(self) -> RunIdentity:
        return self._run

    def monotonic(self) -> float:
        self.clock += 1.0
        return self.clock

    def restart_sandbox(self, fault: FaultSpec) -> JsonObject:
        self.calls.append("restart_sandbox")
        if self.fail_restart:
            raise RuntimeError("restart failed")
        if self.emit_effects:
            self.ledger.record_commit(
                operation_id="publish-1",
                payload={"message": "hello"},
                monotonic_seconds=self.monotonic(),
            )
        return {"boundary": "fake-agent-server", "mode": fault.parameters["mode"]}

    def drop_dispatch_response(self, fault: FaultSpec) -> JsonObject:
        self.calls.append("drop_dispatch_response")
        return {"boundary": "tool-result-before-observation"}

    def sigkill_mid_tool_call(self, fault: FaultSpec) -> JsonObject:
        self.calls.append("sigkill_mid_tool_call")
        return {"target": "fake-tool-process", "signal": "SIGKILL"}

    def partition_network(self, fault: FaultSpec) -> JsonObject:
        self.calls.append("partition_network")
        return {"endpoint": "fake-agent-server", "direction": "both"}

    def heal_network(self, fault: FaultSpec) -> JsonObject:
        self.calls.append("heal_network")
        return {"endpoint": "fake-agent-server"}

    def execute(
        self,
        on_event: Callable[[EventRecord], None],
    ) -> RunObservation:
        if self.emit_effects:
            self.ledger.record_intent(
                operation_id="publish-1",
                payload={"message": "hello"},
                monotonic_seconds=self.monotonic(),
            )
        if self.baseline and self.emit_effects:
            self.ledger.record_commit(
                operation_id="publish-1",
                payload={"message": "hello"},
                monotonic_seconds=self.monotonic(),
            )
        on_event(_event("action-1", ordinal=1))
        if not self.baseline and self.duplicate_effect and self.emit_effects:
            self.ledger.record_commit(
                operation_id="publish-1",
                payload={"message": "hello"},
                monotonic_seconds=self.monotonic(),
            )
        on_event(
            EventRecord(
                event_id="observation-1",
                event_type="ObservationEvent",
                monotonic_seconds=self.monotonic(),
                tool_call_id="tool-1",
            )
        )
        receipts = (
            ()
            if self.baseline or not self.emit_recovery
            else (
                RecoveryReceipt(
                    run_id=self.run.run_id,
                    receipt_type="conversation_history_restored",
                    succeeded=True,
                    monotonic_seconds=self.monotonic(),
                ),
                RecoveryReceipt(
                    run_id=self.run.run_id,
                    receipt_type="runtime_reattached",
                    succeeded=True,
                    monotonic_seconds=self.monotonic(),
                ),
            )
        )
        return RunObservation(
            task_passed=True,
            metrics=RunMetrics(
                wall_time_seconds=10.0 if self.baseline else 15.0,
                recovery_time_seconds=None if self.baseline else 3.0,
                iterations=2 if self.baseline else 3,
                event_count=2,
                tool_calls=1,
                tokens=100,
                cost_usd=0.01,
            ),
            recovery_receipts=receipts,
        )


def _scenario() -> Scenario:
    event_trigger = EventTrigger(
        event_type="ActionEvent",
        ordinal=1,
    )
    return Scenario(
        schema_version=1,
        scenario_id="e2e-faults",
        benchmark="swebench",
        instance_id="project__issue-1",
        agent_config="default",
        seed=23,
        faults=(
            FaultSpec(
                fault_id="restart",
                kind=FaultKind.SANDBOX_RESTART,
                trigger=event_trigger,
                parameters={"mode": "hard"},
            ),
            FaultSpec(
                fault_id="lost-response",
                kind=FaultKind.LOST_DISPATCH_RESPONSE,
                trigger=event_trigger,
            ),
            FaultSpec(
                fault_id="sigkill",
                kind=FaultKind.SIGKILL_MID_TOOL_CALL,
                trigger=event_trigger,
            ),
            FaultSpec(
                fault_id="partition",
                kind=FaultKind.NETWORK_PARTITION,
                trigger=event_trigger,
            ),
        ),
    )


def _event(event_id: str, *, ordinal: int) -> EventRecord:
    return EventRecord(
        event_id=event_id,
        event_type="ActionEvent",
        monotonic_seconds=float(ordinal),
        tool_call_id="tool-1",
    )
