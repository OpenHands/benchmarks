"""Deterministic schedule compilation and persisted-event trigger matching."""

import random
from collections import Counter
from collections.abc import Iterable

from benchmarks.reliability.models import EventRecord, FaultSpec, Scenario


class FaultSchedule:
    """Stateful deterministic schedule for one scenario run."""

    def __init__(self, scenario: Scenario) -> None:
        self._faults = {fault.fault_id: fault for fault in scenario.faults}
        fault_ids = sorted(self._faults)
        random.Random(scenario.seed).shuffle(fault_ids)
        self._tie_break_order = {
            fault_id: rank for rank, fault_id in enumerate(fault_ids)
        }
        self._event_counts: Counter[str] = Counter()
        self._seen_event_ids: set[str] = set()
        self._fired_fault_ids: set[str] = set()

    def pending(self) -> tuple[FaultSpec, ...]:
        """Return faults that have not fired in deterministic order."""
        return tuple(
            fault
            for fault in self._ordered_faults()
            if fault.fault_id not in self._fired_fault_ids
        )

    def fired(self) -> tuple[FaultSpec, ...]:
        """Return faults that have already fired in deterministic order."""
        return tuple(
            fault
            for fault in self._ordered_faults()
            if fault.fault_id in self._fired_fault_ids
        )

    def observe(self, events: Iterable[EventRecord]) -> tuple[FaultSpec, ...]:
        """Return newly matched faults for previously unseen persisted events."""
        matched: list[FaultSpec] = []
        for event in events:
            if event.event_id in self._seen_event_ids:
                continue
            self._seen_event_ids.add(event.event_id)
            self._event_counts[event.event_type] += 1
            ordinal = self._event_counts[event.event_type]

            event_matches = [
                fault for fault in self.pending() if _matches(fault, event, ordinal)
            ]
            event_matches.sort(key=lambda item: self._tie_break_order[item.fault_id])
            for fault in event_matches:
                self._fired_fault_ids.add(fault.fault_id)
                matched.append(fault)
        return tuple(matched)

    def _ordered_faults(self) -> tuple[FaultSpec, ...]:
        return tuple(
            sorted(
                self._faults.values(),
                key=lambda fault: (
                    fault.trigger.ordinal,
                    fault.trigger.event_type,
                    self._tie_break_order[fault.fault_id],
                ),
            )
        )


def compile_schedule(scenario: Scenario) -> FaultSchedule:
    """Validate and compile a scenario into a deterministic fault schedule."""
    return FaultSchedule(scenario)


def _matches(fault: FaultSpec, event: EventRecord, ordinal: int) -> bool:
    trigger = fault.trigger
    return (
        trigger.event_type == event.event_type
        and trigger.ordinal == ordinal
        and (trigger.tool_call_id is None or trigger.tool_call_id == event.tool_call_id)
    )
