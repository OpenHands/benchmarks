"""Deterministic, inspectable reliability graders."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from benchmarks.reliability.artifacts import (
    metrics_from_json,
    read_json,
    read_jsonl,
)
from benchmarks.reliability.models import (
    GraderResult,
    JsonObject,
    JsonValue,
    Scenario,
)


@dataclass(frozen=True, slots=True)
class GradingEvidence:
    """Artifact references supplied to deterministic graders."""

    scenario: Scenario
    task_result_path: Path
    events_path: Path
    fault_receipts_path: Path
    recovery_receipts_path: Path
    effects_path: Path
    metrics_path: Path
    baseline_metrics_path: Path | None


def grade_completion_resume(evidence: GradingEvidence) -> GraderResult:
    """Judge task completion and genuine resume from explicit receipts."""
    task = read_json(evidence.task_result_path)
    faults = read_jsonl(evidence.fault_receipts_path)
    recovery = read_jsonl(evidence.recovery_receipts_path)
    reasons: list[str] = []

    task_passed = task.get("task_passed") is True
    if not task_passed:
        reasons.append("task_failed")
    if task.get("fresh_retry_used") is True:
        reasons.append("fresh_retry_used")

    applied_ids = {
        _required_string(receipt, "fault_id")
        for receipt in faults
        if receipt.get("status") == "applied"
    }
    failed_ids = {
        _required_string(receipt, "fault_id")
        for receipt in faults
        if receipt.get("status") == "failed"
    }
    expected_ids = {fault.fault_id for fault in evidence.scenario.faults}
    missing_ids = sorted(expected_ids - applied_ids)
    if missing_ids:
        reasons.append("fault_not_injected")
    if failed_ids:
        reasons.append("fault_injection_failed")

    successful_recovery = {
        _required_string(receipt, "receipt_type")
        for receipt in recovery
        if receipt.get("succeeded") is True
    }
    required_recovery = {
        "conversation_history_restored",
        "runtime_reattached",
    }
    missing_recovery = sorted(required_recovery - successful_recovery)
    if evidence.scenario.faults and missing_recovery:
        reasons.append("recovery_receipt_missing")

    passed = not reasons
    value: JsonObject = {
        "score": 1 if passed else 0,
        "task_passed": task_passed,
        "expected_faults": len(expected_ids),
        "applied_faults": len(applied_ids & expected_ids),
        "missing_fault_ids": _json_strings(missing_ids),
        "failed_fault_ids": _json_strings(sorted(failed_ids)),
        "missing_recovery_receipts": _json_strings(missing_recovery),
    }
    return GraderResult(
        grader="completion_resume",
        passed=passed,
        value=value,
        reason_codes=tuple(reasons or ["completed_and_resumed"]),
        evidence_refs=(
            str(evidence.task_result_path),
            str(evidence.fault_receipts_path),
            str(evidence.recovery_receipts_path),
            str(evidence.events_path),
        ),
    )


def grade_no_duplicate_effect(evidence: GradingEvidence) -> GraderResult:
    """Judge committed effect counts using the scenario-owned ledger."""
    records = read_jsonl(evidence.effects_path)
    intents: dict[str, list[JsonObject]] = defaultdict(list)
    commits: dict[str, list[JsonObject]] = defaultdict(list)
    for record in records:
        operation_id = _required_string(record, "operation_id")
        phase = _required_string(record, "phase")
        if phase == "intent":
            intents[operation_id].append(record)
        elif phase == "commit":
            commits[operation_id].append(record)
        else:
            raise ValueError(f"unknown effect phase: {phase}")

    missing: list[str] = []
    duplicated: dict[str, int] = {}
    payload_mismatch: list[str] = []
    unexpected = sorted(set(commits) - set(intents))

    for operation_id, intent_records in intents.items():
        expected = len(intent_records)
        actual = len(commits.get(operation_id, ()))
        if actual < expected:
            missing.append(operation_id)
        if actual > expected:
            duplicated[operation_id] = actual - expected

        intent_digests = Counter(
            _required_string(item, "payload_digest") for item in intent_records
        )
        commit_digests = Counter(
            _required_string(item, "payload_digest")
            for item in commits.get(operation_id, ())
        )
        if any(
            commit_digests[digest] < count for digest, count in intent_digests.items()
        ):
            payload_mismatch.append(operation_id)

    passed = bool(records) and not (
        missing or duplicated or unexpected or payload_mismatch
    )
    if not records:
        reason_codes = ("effect_evidence_missing",)
    else:
        reason_codes = tuple(
            reason
            for condition, reason in (
                (missing, "effect_missing"),
                (duplicated, "duplicate_effect"),
                (unexpected, "unexpected_effect"),
                (payload_mismatch, "effect_payload_mismatch"),
            )
            if condition
        ) or ("effects_exactly_once",)

    duplicated_json: JsonObject = {
        operation_id: count for operation_id, count in duplicated.items()
    }
    value: JsonObject = {
        "score": 1 if passed else 0,
        "logical_operations": len(intents),
        "missing_operation_ids": _json_strings(sorted(missing)),
        "duplicated_operations": duplicated_json,
        "unexpected_operation_ids": _json_strings(unexpected),
        "payload_mismatch_operation_ids": _json_strings(sorted(payload_mismatch)),
        "excess_commits": sum(duplicated.values()),
    }
    return GraderResult(
        grader="no_duplicate_effect",
        passed=passed,
        value=value,
        reason_codes=reason_codes,
        evidence_refs=(str(evidence.effects_path),),
    )


def grade_recovery_overhead(evidence: GradingEvidence) -> GraderResult:
    """Compare faulted metrics with the matched no-fault baseline."""
    if evidence.baseline_metrics_path is None:
        return GraderResult(
            grader="recovery_overhead",
            passed=False,
            value=None,
            reason_codes=("baseline_missing",),
            evidence_refs=(str(evidence.metrics_path),),
        )

    faulted = metrics_from_json(evidence.metrics_path)
    baseline = metrics_from_json(evidence.baseline_metrics_path)
    value: JsonObject = {
        "wall_time_seconds": faulted.wall_time_seconds,
        "baseline_wall_time_seconds": baseline.wall_time_seconds,
        "wall_time_delta_seconds": (
            faulted.wall_time_seconds - baseline.wall_time_seconds
        ),
        "wall_time_ratio": _ratio(
            faulted.wall_time_seconds,
            baseline.wall_time_seconds,
        ),
        "recovery_time_seconds": faulted.recovery_time_seconds,
        "iteration_delta": faulted.iterations - baseline.iterations,
        "event_delta": faulted.event_count - baseline.event_count,
        "tool_call_delta": faulted.tool_calls - baseline.tool_calls,
        "token_delta": faulted.tokens - baseline.tokens,
        "cost_delta_usd": faulted.cost_usd - baseline.cost_usd,
    }
    return GraderResult(
        grader="recovery_overhead",
        passed=True,
        value=value,
        reason_codes=("baseline_compared",),
        evidence_refs=(
            str(evidence.metrics_path),
            str(evidence.baseline_metrics_path),
        ),
    )


def grade_all(evidence: GradingEvidence) -> tuple[GraderResult, ...]:
    """Run every deterministic grader over one artifact set."""
    return (
        grade_completion_resume(evidence),
        grade_no_duplicate_effect(evidence),
        grade_recovery_overhead(evidence),
    )


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _required_string(value: JsonObject, field_name: str) -> str:
    item = value.get(field_name)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{field_name} must be a non-empty string")
    return item


def _json_strings(values: list[str]) -> list[JsonValue]:
    result: list[JsonValue] = []
    result.extend(values)
    return result
