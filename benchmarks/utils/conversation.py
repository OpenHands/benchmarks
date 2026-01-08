from __future__ import annotations

import base64
from typing import Callable

from openhands.sdk import Event, get_logger
from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)

ConversationCallback = Callable[[Event], None]


def build_event_persistence_callback(
    workspace: RemoteWorkspace, instance_id: str
) -> tuple[ConversationCallback, str]:
    """
    Create a callback that appends serialized events to a JSONL file in the runtime.

    Args:
        workspace: Remote workspace backing the conversation.
        instance_id: Identifier for the evaluation instance.

    Returns:
        A tuple of (callback, conversation_file_path).
    """
    workspace_dir_raw = getattr(workspace, "working_dir", None)
    workspace_dir = (
        str(workspace_dir_raw) if workspace_dir_raw else "/workspace"
    ).rstrip("/") or "/workspace"
    conversations_dir = f"{workspace_dir}/conversations"
    conversation_file = f"{conversations_dir}/{instance_id}.jsonl"

    def _persist_event(event: Event) -> None:
        try:
            serialized = event.model_dump_json(exclude_none=True)
        except Exception as exc:  # best-effort; never block the run
            logger.debug(
                "Skipping persistence for %s; serialization failed: %s",
                instance_id,
                exc,
            )
            return

        try:
            encoded = base64.b64encode(serialized.encode("utf-8")).decode("ascii")
            command = (
                "python - <<'PY'\n"
                "import base64\n"
                "from pathlib import Path\n"
                f"path = Path('{conversation_file}')\n"
                "path.parent.mkdir(parents=True, exist_ok=True)\n"
                f"data = base64.b64decode('{encoded}')\n"
                "with path.open('ab') as f:\n"
                "    f.write(data + b'\\n')\n"
                "PY"
            )
            result = workspace.execute_command(command)
            if result.exit_code != 0:
                logger.debug(
                    "Event persistence command failed for %s (exit_code=%s): %s",
                    instance_id,
                    result.exit_code,
                    result.stderr,
                )
        except Exception as exc:
            logger.debug(
                "Failed to persist conversation event for %s: %s", instance_id, exc
            )

    return _persist_event, conversation_file
