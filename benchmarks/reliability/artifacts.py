"""Stable JSON/JSONL artifact storage for reliability runs."""

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from benchmarks.reliability.models import (
    EffectRecord,
    EventRecord,
    FaultReceipt,
    JsonObject,
    JsonValue,
    RecoveryReceipt,
    ReliabilityResult,
    RunIdentity,
    RunMetrics,
    RunObservation,
    Scenario,
    as_json_object,
)


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Canonical artifact paths and append operations for one run."""

    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def fault_receipts_path(self) -> Path:
        return self.root / "fault_receipts.jsonl"

    @property
    def recovery_receipts_path(self) -> Path:
        return self.root / "recovery_receipts.jsonl"

    @property
    def effects_path(self) -> Path:
        return self.root / "effects.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    @property
    def task_result_path(self) -> Path:
        return self.root / "task_result.json"

    @property
    def reliability_result_path(self) -> Path:
        return self.root / "reliability_result.json"

    def initialize(
        self,
        *,
        run: RunIdentity,
        scenario: Scenario,
        baseline: bool,
    ) -> None:
        """Create an empty, self-identifying artifact directory."""
        self.root.mkdir(parents=True, exist_ok=False)
        write_json(
            self.manifest_path,
            {
                "run": as_json_object(run),
                "scenario": as_json_object(scenario),
                "baseline": baseline,
            },
        )
        for path in (
            self.events_path,
            self.fault_receipts_path,
            self.recovery_receipts_path,
            self.effects_path,
        ):
            path.touch(exist_ok=False)

    def append_event(self, event: EventRecord) -> None:
        append_jsonl(self.events_path, as_json_object(event))

    def append_fault_receipt(self, receipt: FaultReceipt) -> None:
        append_jsonl(self.fault_receipts_path, as_json_object(receipt))

    def append_recovery_receipt(self, receipt: RecoveryReceipt) -> None:
        append_jsonl(self.recovery_receipts_path, as_json_object(receipt))

    def append_effect(self, effect: EffectRecord) -> None:
        append_jsonl(self.effects_path, as_json_object(effect))

    def write_observation(self, observation: RunObservation) -> None:
        write_json(self.metrics_path, as_json_object(observation.metrics))
        write_json(
            self.task_result_path,
            {
                "task_passed": observation.task_passed,
                "fresh_retry_used": observation.fresh_retry_used,
                "task_result": observation.task_result,
            },
        )
        for receipt in observation.recovery_receipts:
            self.append_recovery_receipt(receipt)

    def write_result(self, result: ReliabilityResult) -> None:
        write_json(self.reliability_result_path, as_json_object(result))


_append_locks_guard = threading.Lock()
_append_locks: dict[Path, threading.Lock] = {}


def append_jsonl(path: Path, value: JsonObject) -> None:
    """Append one durable, newline-delimited JSON object under a path lock."""
    with _append_locks_guard:
        lock = _append_locks.setdefault(path.resolve(), threading.Lock())
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    with lock, path.open("a", encoding="utf-8") as stream:
        stream.write(payload)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path: Path, value: JsonObject) -> None:
    """Atomically replace one JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> JsonObject:
    """Read one JSON object and reject non-object payloads."""
    value: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(JsonObject, value)


def read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    """Read JSONL objects while rejecting blank or non-object records."""
    records: list[JsonObject] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is blank")
            value: JsonValue = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            records.append(cast(JsonObject, value))
    return tuple(records)


def metrics_from_json(path: Path) -> RunMetrics:
    """Load the strict subset of raw metrics used by reliability grading."""
    value = read_json(path)
    recovery_raw = value.get("recovery_time_seconds")
    return RunMetrics(
        wall_time_seconds=_number(value, "wall_time_seconds"),
        recovery_time_seconds=(
            None if recovery_raw is None else _number(value, "recovery_time_seconds")
        ),
        iterations=_integer(value, "iterations"),
        event_count=_integer(value, "event_count"),
        tool_calls=_integer(value, "tool_calls"),
        tokens=_integer(value, "tokens"),
        cost_usd=_number(value, "cost_usd"),
    )


def _number(value: JsonObject, field_name: str) -> float:
    item = value.get(field_name)
    if isinstance(item, bool) or not isinstance(item, int | float):
        raise ValueError(f"{field_name} must be numeric")
    return float(item)


def _integer(value: JsonObject, field_name: str) -> int:
    item = value.get(field_name)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"{field_name} must be an integer")
    return item
