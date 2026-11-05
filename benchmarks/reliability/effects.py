"""Scenario-owned external-effect ledger and deterministic query helpers."""

import hashlib
import json
from pathlib import Path

from benchmarks.reliability.artifacts import append_jsonl, read_jsonl
from benchmarks.reliability.models import EffectRecord, JsonObject, JsonValue


class EffectLedger:
    """Append-only evidence independent of the agent event log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def record_intent(
        self,
        *,
        operation_id: str,
        payload: JsonValue,
        monotonic_seconds: float,
        idempotency_key: str | None = None,
    ) -> EffectRecord:
        """Record one intended logical external operation."""
        return self._append(
            operation_id=operation_id,
            phase="intent",
            payload=payload,
            monotonic_seconds=monotonic_seconds,
            idempotency_key=idempotency_key,
        )

    def record_commit(
        self,
        *,
        operation_id: str,
        payload: JsonValue,
        monotonic_seconds: float,
        idempotency_key: str | None = None,
    ) -> EffectRecord:
        """Record one externally committed effect."""
        return self._append(
            operation_id=operation_id,
            phase="commit",
            payload=payload,
            monotonic_seconds=monotonic_seconds,
            idempotency_key=idempotency_key,
        )

    def records(self) -> tuple[JsonObject, ...]:
        """Return every intent and commit record in append order."""
        return read_jsonl(self.path)

    def _append(
        self,
        *,
        operation_id: str,
        phase: str,
        payload: JsonValue,
        monotonic_seconds: float,
        idempotency_key: str | None,
    ) -> EffectRecord:
        if not operation_id:
            raise ValueError("operation_id must not be empty")
        record = EffectRecord(
            operation_id=operation_id,
            phase=phase,
            payload_digest=payload_digest(payload),
            monotonic_seconds=monotonic_seconds,
            idempotency_key=idempotency_key,
        )
        append_jsonl(
            self.path,
            {
                "operation_id": record.operation_id,
                "phase": record.phase,
                "payload_digest": record.payload_digest,
                "monotonic_seconds": record.monotonic_seconds,
                "idempotency_key": record.idempotency_key,
            },
        )
        return record


def payload_digest(payload: JsonValue) -> str:
    """Return a stable SHA-256 digest for one JSON-compatible payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
