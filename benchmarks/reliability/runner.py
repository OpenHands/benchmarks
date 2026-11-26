"""Baseline/faulted orchestration for reliability benchmark adapters."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from benchmarks.reliability.artifacts import RunArtifacts, read_json
from benchmarks.reliability.grading import (
    GradingEvidence,
    grade_completion_resume,
    grade_no_duplicate_effect,
    grade_recovery_overhead,
)
from benchmarks.reliability.injectors import FaultContext, FaultController
from benchmarks.reliability.models import (
    EventRecord,
    ReliabilityResult,
    RunIdentity,
    RunObservation,
    Scenario,
)
from benchmarks.reliability.schedule import compile_schedule


class RunSession(FaultContext, Protocol):
    """One opened baseline or faulted run from a benchmark adapter."""

    def execute(
        self,
        on_event: Callable[[EventRecord], None],
    ) -> RunObservation:
        """Execute the task and synchronously publish persisted events."""
        ...


class ReliabilityAdapter(Protocol):
    """Bridge one real benchmark/runtime to the generic reliability harness."""

    def open_run(
        self,
        *,
        scenario: Scenario,
        run: RunIdentity,
        artifacts: RunArtifacts,
        baseline: bool,
    ) -> RunSession:
        """Prepare one independent run with explicit fault capabilities."""
        ...


ResultSink = Callable[[ReliabilityResult], None]


def run_scenario(
    scenario: Scenario,
    *,
    adapter: ReliabilityAdapter,
    output_root: Path,
    on_result: ResultSink | None = None,
) -> ReliabilityResult:
    """Execute a matched baseline and faulted run, then grade the outcome."""
    scenario_root = (
        output_root / _safe_component(scenario.scenario_id) / str(scenario.seed)
    )
    baseline_run = _run_identity(scenario, scenario_root, "baseline")
    faulted_run = _run_identity(scenario, scenario_root, "faulted")

    baseline_artifacts = _execute_run(
        scenario=scenario,
        run=baseline_run,
        adapter=adapter,
        baseline=True,
    )
    faulted_artifacts = _execute_run(
        scenario=scenario,
        run=faulted_run,
        adapter=adapter,
        baseline=False,
    )
    evidence = GradingEvidence(
        scenario=scenario,
        task_result_path=faulted_artifacts.task_result_path,
        events_path=faulted_artifacts.events_path,
        fault_receipts_path=faulted_artifacts.fault_receipts_path,
        recovery_receipts_path=faulted_artifacts.recovery_receipts_path,
        effects_path=faulted_artifacts.effects_path,
        metrics_path=faulted_artifacts.metrics_path,
        baseline_metrics_path=baseline_artifacts.metrics_path,
    )
    task_passed = (
        read_json(faulted_artifacts.task_result_path).get("task_passed") is True
    )
    result = ReliabilityResult(
        run=faulted_run,
        task_passed=task_passed,
        completion_resume=grade_completion_resume(evidence),
        no_duplicate_effect=grade_no_duplicate_effect(evidence),
        recovery_overhead=grade_recovery_overhead(evidence),
    )
    faulted_artifacts.write_result(result)
    if on_result is not None:
        on_result(result)
    return result


def _execute_run(
    *,
    scenario: Scenario,
    run: RunIdentity,
    adapter: ReliabilityAdapter,
    baseline: bool,
) -> RunArtifacts:
    artifacts = RunArtifacts(run.artifact_dir)
    artifacts.initialize(run=run, scenario=scenario, baseline=baseline)
    session = adapter.open_run(
        scenario=scenario,
        run=run,
        artifacts=artifacts,
        baseline=baseline,
    )
    controller = (
        None
        if baseline
        else FaultController(
            schedule=compile_schedule(scenario),
            context=session,
            on_receipt=artifacts.append_fault_receipt,
        )
    )

    def on_event(event: EventRecord) -> None:
        artifacts.append_event(event)
        if controller is not None:
            controller.observe(event)

    try:
        observation = session.execute(on_event)
    finally:
        if controller is not None:
            controller.release_all()
    artifacts.write_observation(observation)
    return artifacts


def _run_identity(
    scenario: Scenario,
    scenario_root: Path,
    variant: str,
) -> RunIdentity:
    run_id = f"{scenario.scenario_id}:{scenario.seed}:{variant}"
    return RunIdentity(
        run_id=run_id,
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        artifact_dir=scenario_root / variant,
    )


def _safe_component(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
    if safe in {"", ".", ".."}:
        raise ValueError("scenario_id does not produce a safe artifact path")
    return safe
