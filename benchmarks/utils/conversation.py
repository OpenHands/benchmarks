from __future__ import annotations

from typing import Any, Callable

from openhands.sdk import Event, get_logger


logger = get_logger(__name__)

ConversationCallback = Callable[[Event], None]

# Max size for full event logging (256KB). Larger events log metadata only.
MAX_EVENT_SIZE_BYTES = 256 * 1024


def _extract_event_metadata(event: Event) -> dict[str, Any]:
    """Extract metadata from an event for logging without full content."""
    metadata: dict[str, Any] = {
        "event_type": type(event).__name__,
    }

    # Extract common fields if present
    if hasattr(event, "id"):
        metadata["id"] = event.id
    if hasattr(event, "timestamp"):
        metadata["timestamp"] = str(event.timestamp)
    if hasattr(event, "source"):
        metadata["source"] = str(event.source)

    # Extract tool-specific metadata
    if hasattr(event, "tool_name"):
        metadata["tool_name"] = event.tool_name
    if hasattr(event, "tool_call_id"):
        metadata["tool_call_id"] = event.tool_call_id

    # For observations, extract key fields without full content
    if hasattr(event, "observation"):
        obs = event.observation
        if hasattr(obs, "command"):
            metadata["command"] = _truncate(str(obs.command), 500)
        if hasattr(obs, "path"):
            metadata["path"] = obs.path
        if hasattr(obs, "exit_code"):
            metadata["exit_code"] = obs.exit_code
        if hasattr(obs, "is_error"):
            metadata["is_error"] = obs.is_error

    # For actions, extract key fields
    if hasattr(event, "action"):
        action = event.action
        if hasattr(action, "command"):
            metadata["command"] = _truncate(str(action.command), 500)
        if hasattr(action, "path"):
            metadata["path"] = action.path
        if hasattr(action, "thought"):
            metadata["thought"] = _truncate(str(action.thought), 500)

    return metadata


def _truncate(s: str, max_len: int) -> str:
    """Truncate string with ellipsis if too long."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def build_event_persistence_callback(
    run_id: str, instance_id: str
) -> ConversationCallback:
    """
    Create a callback that logs events for later retrieval.

    Small events are logged in full; large events log metadata only to avoid
    size limits and ensure logs persist beyond pod lifetime.

    Args:
        run_id: Unique identifier for this evaluation run (e.g., job name).
        instance_id: Identifier for the evaluation instance.

    Returns:
        A callback function to be passed to Conversation.
    """

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
