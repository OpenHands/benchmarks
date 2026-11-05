"""Typed public data contracts for reliability scenarios and results."""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast


type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class FaultKind(StrEnum):
    """Supported fault categories."""

    SANDBOX_RESTART = "sandbox_restart"
    LOST_DISPATCH_RESPONSE = "lost_dispatch_response"
    SIGKILL_MID_TOOL_CALL = "sigkill_mid_tool_call"
    NETWORK_PARTITION = "network_partition"


class ReceiptStatus(StrEnum):
    """Outcome of a requested fault or recovery operation."""

    APPLIED = "applied"
    RELEASED = "released"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EventTrigger:
    """Select a deterministic persisted-event occurrence."""

    event_type: str
    ordinal: int
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if not self.event_type:
            raise ValueError("event_type must not be empty")
        if self.ordinal < 1:
            raise ValueError("ordinal must be at least 1")


@dataclass(frozen=True, slots=True)
class FaultSpec:
    """One compiled fault in a scenario schedule."""

    fault_id: str
    kind: FaultKind
    trigger: EventTrigger
    parameters: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fault_id:
            raise ValueError("fault_id must not be empty")


@dataclass(frozen=True, slots=True)
class Scenario:
    """Versioned, declarative benchmark scenario."""

    schema_version: int
    scenario_id: str
    benchmark: str
    instance_id: str
    agent_config: str
    seed: int
    faults: tuple[FaultSpec, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported schema_version {self.schema_version}; expected 1"
            )
        for field_name in (
            "scenario_id",
            "benchmark",
            "instance_id",
            "agent_config",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        fault_ids = [fault.fault_id for fault in self.faults]
        if len(fault_ids) != len(set(fault_ids)):
            raise ValueError("fault_id values must be unique within a scenario")


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Stable identity and artifact root for one scenario run."""

    run_id: str
    scenario_id: str
    seed: int
    artifact_dir: Path


@dataclass(frozen=True, slots=True)
class EventRecord:
    """Minimal persisted event record consumed by deterministic schedules."""

    event_id: str
    event_type: str
    monotonic_seconds: float
    tool_call_id: str | None = None
    details: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not self.event_type:
            raise ValueError("event_type must not be empty")


@dataclass(frozen=True, slots=True)
class FaultReceipt:
    """Inspectable evidence that a scheduled fault operation ran."""

    run_id: str
    fault_id: str
    kind: FaultKind
    status: ReceiptStatus
    monotonic_seconds: float
    event_id: str | None
    tool_call_id: str | None
    details: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecoveryReceipt:
    """Inspectable evidence produced by the product's recovery path."""

    run_id: str
    receipt_type: str
    succeeded: bool
    monotonic_seconds: float
    details: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EffectRecord:
    """Intent or commit recorded by a scenario-owned external-effect ledger."""

    operation_id: str
    phase: str
    payload_digest: str
    monotonic_seconds: float
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in {"intent", "commit"}:
            raise ValueError("effect phase must be 'intent' or 'commit'")


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Raw metrics used to compare faulted and no-fault runs."""

    wall_time_seconds: float
    recovery_time_seconds: float | None = None
    iterations: int = 0
    event_count: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.wall_time_seconds < 0:
            raise ValueError("wall_time_seconds must not be negative")


@dataclass(frozen=True, slots=True)
class RunObservation:
    """Adapter-produced terminal result and recovery evidence."""

    task_passed: bool
    metrics: RunMetrics
    recovery_receipts: tuple[RecoveryReceipt, ...] = ()
    fresh_retry_used: bool = False
    task_result: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraderResult:
    """One inspectable grader result."""

    grader: str
    passed: bool
    value: JsonValue
    reason_codes: tuple[str, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReliabilityResult:
    """Combined task and reliability outcome for one run."""

    run: RunIdentity
    task_passed: bool
    completion_resume: GraderResult
    no_duplicate_effect: GraderResult
    recovery_overhead: GraderResult


def as_json_object(value: object) -> JsonObject:
    """Convert one dataclass tree to a JSON-compatible mapping."""
    raw = asdict(cast(Any, value))
    normalized = _normalize_json(raw)
    if not isinstance(normalized, dict):
        raise TypeError("dataclass did not serialize to a JSON object")
    return cast(JsonObject, normalized)


def _normalize_json(value: object) -> JsonValue:
    if isinstance(value, StrEnum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_json(item) for item in value]
    raise TypeError(f"cannot serialize {type(value).__name__} as JSON")
