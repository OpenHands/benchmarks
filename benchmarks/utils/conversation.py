from __future__ import annotations

import os
from typing import Any, Callable

from openhands.sdk import Event, get_logger


logger = get_logger(__name__)

ConversationCallback = Callable[[Event], None]

# Max size for full event logging (256KB). Larger events log metadata only.
MAX_EVENT_SIZE_BYTES = 256 * 1024
CONVERSATION_EVENT_LOGGING_ENV_VAR = "ENABLE_CONVERSATION_EVENT_LOGGING"


def _extract_event_metadata(event: Event) -> dict[str, Any]:
    """Extract metadata from an event for logging without full content."""
    metadata: dict[str, Any] = {
        "event_type": type(event).__name__,
    }

    # Extract common fields if present
    if hasattr(event, "id"):
        metadata["id"] = getattr(event, "id")
    if hasattr(event, "timestamp"):
        metadata["timestamp"] = str(getattr(event, "timestamp"))
    if hasattr(event, "source"):
        metadata["source"] = str(getattr(event, "source"))

    # Extract tool-specific metadata
    if hasattr(event, "tool_name"):
        metadata["tool_name"] = getattr(event, "tool_name")
    if hasattr(event, "tool_call_id"):
        metadata["tool_call_id"] = getattr(event, "tool_call_id")

    # For observations, extract key fields without full content
    if hasattr(event, "observation"):
        obs = getattr(event, "observation")
        if hasattr(obs, "command"):
            metadata["command"] = _truncate(str(getattr(obs, "command")), 500)
        if hasattr(obs, "path"):
            metadata["path"] = getattr(obs, "path")
        if hasattr(obs, "exit_code"):
            metadata["exit_code"] = getattr(obs, "exit_code")
        if hasattr(obs, "is_error"):
            metadata["is_error"] = getattr(obs, "is_error")

    # For actions, extract key fields
    if hasattr(event, "action"):
        action = getattr(event, "action")
        if hasattr(action, "command"):
            metadata["command"] = _truncate(str(getattr(action, "command")), 500)
        if hasattr(action, "path"):
            metadata["path"] = getattr(action, "path")
        if hasattr(action, "thought"):
            metadata["thought"] = _truncate(str(getattr(action, "thought")), 500)

    return metadata


def _truncate(s: str, max_len: int) -> str:
    """Truncate string with ellipsis if too long."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def build_event_persistence_callback(
    run_id: str, instance_id: str, attempt: int = 1
) -> ConversationCallback:
    """
    Create a callback that logs events for later retrieval.

    Small events are logged in full; large events log metadata only to avoid
    size limits and ensure logs persist beyond pod lifetime.
    Logging is enabled by default; set ENABLE_CONVERSATION_EVENT_LOGGING to a
    falsey value to disable.

    Args:
        run_id: Unique identifier for this evaluation run (e.g., job name).
        instance_id: Identifier for the evaluation instance.
        attempt: Attempt number for retries (1-indexed).

    Returns:
        A callback function to be passed to Conversation.
    """
    if not bool(os.environ.get(CONVERSATION_EVENT_LOGGING_ENV_VAR, True)):
        return lambda event: None

    def _persist_event(event: Event) -> None:
        try:
            serialized = event.model_dump_json(exclude_none=True)
            event_size = len(serialized.encode("utf-8"))

            if event_size <= MAX_EVENT_SIZE_BYTES:
                # Small event: log full content
                logger.info(
                    "conversation_event",
                    extra={
                        "run_id": run_id,
                        "instance_id": instance_id,
                        "attempt": attempt,
                        "event_type": type(event).__name__,
                        "event_size": event_size,
                        "event": serialized,
                    },
                )
            else:
                # Large event: log metadata only
                metadata = _extract_event_metadata(event)
                logger.info(
                    "conversation_event_metadata",
                    extra={
                        "run_id": run_id,
                        "instance_id": instance_id,
                        "attempt": attempt,
                        "event_size": event_size,
                        "truncated": True,
                        **metadata,
                    },
                )
        except Exception as exc:
            # Best-effort; never block the run
            logger.debug(
                "Failed to persist conversation event for %s: %s", instance_id, exc
            )

    return _persist_event
