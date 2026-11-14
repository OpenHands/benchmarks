"""Inspectable, adapter-backed fault-injection reliability benchmark."""

from benchmarks.reliability.models import (
    EffectRecord,
    EventRecord,
    FaultKind,
    FaultReceipt,
    FaultSpec,
    ReliabilityResult,
    Scenario,
)
from benchmarks.reliability.runner import ReliabilityAdapter, run_scenario


__all__ = [
    "EffectRecord",
    "EventRecord",
    "FaultKind",
    "FaultReceipt",
    "FaultSpec",
    "ReliabilityAdapter",
    "ReliabilityResult",
    "Scenario",
    "run_scenario",
]
