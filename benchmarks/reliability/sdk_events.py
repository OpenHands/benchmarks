"""Lossless trigger metadata conversion for OpenHands SDK events."""

import time
from collections.abc import Callable

from benchmarks.reliability.models import EventRecord, JsonObject
from openhands.sdk.event import (
    ACPToolCallEvent,
    ActionEvent,
    Event,
    ObservationBaseEvent,
)


SDKEventCallback = Callable[[Event], None]
ReliabilityEventCallback = Callable[[EventRecord], None]


def sdk_event_to_record(
    event: Event,
    *,
    monotonic_seconds: float | None = None,
) -> EventRecord:
    """Convert one persisted SDK event without exposing hidden reasoning."""
    tool_call_id: str | None = None
    tool_name: str | None = None
    if isinstance(event, ActionEvent | ObservationBaseEvent | ACPToolCallEvent):
        tool_call_id = str(event.tool_call_id)
    if isinstance(event, ActionEvent | ObservationBaseEvent):
        tool_name = event.tool_name

    details: JsonObject = {
        "source": str(event.source),
        "timestamp": event.timestamp,
    }
    if tool_name is not None:
        details["tool_name"] = tool_name
    return EventRecord(
        event_id=str(event.id),
        event_type=type(event).__name__,
        monotonic_seconds=(
            time.monotonic() if monotonic_seconds is None else monotonic_seconds
        ),
        tool_call_id=tool_call_id,
        details=details,
    )


def build_reliability_event_callback(
    on_event: ReliabilityEventCallback,
    *,
    persistence_callbacks: tuple[SDKEventCallback, ...] = (),
    monotonic: Callable[[], float] = time.monotonic,
) -> SDKEventCallback:
    """Persist through existing callbacks before triggering fault injection.

    Each persistence callback runs first. Only after all return is the event
    exposed to the reliability scheduler. This preserves the required
    persist-before-inject ordering.
    """

    def callback(event: Event) -> None:
        for existing_callback in persistence_callbacks:
            existing_callback(event)
        on_event(
            sdk_event_to_record(
                event,
                monotonic_seconds=monotonic(),
            )
        )

    return callback
