"""Utilities for ACP (Agent Communication Protocol) agent support."""

import json
import os
import tempfile

from openhands.sdk import get_logger
from openhands.sdk.workspace import RemoteWorkspace

logger = get_logger(__name__)

# Mapping of ACP agent types to the API key env vars they require.
_ACP_ENV_VARS: dict[str, str] = {
    "acp-claude": "ANTHROPIC_API_KEY",
    "acp-codex": "OPENAI_API_KEY",
}

# Mapping of ACP agent types to their ACP command.
_ACP_COMMANDS: dict[str, list[str]] = {
    "acp-claude": ["claude-agent-acp"],
    "acp-codex": ["codex-acp"],
}


def is_acp_agent(agent_type: str) -> bool:
    """Return True if *agent_type* refers to an ACP-based agent."""
    return agent_type in _ACP_COMMANDS


def get_acp_command(agent_type: str) -> list[str]:
    """Return the ACP command list for the given *agent_type*.

    Raises ``ValueError`` for unknown ACP agent types.
    """
    try:
        return list(_ACP_COMMANDS[agent_type])
    except KeyError:
        raise ValueError(
            f"Unknown ACP agent type: {agent_type!r}. "
            f"Known types: {list(_ACP_COMMANDS)}"
        )


def get_acp_forward_env(
    agent_type: str, forward_env: list[str] | None = None
) -> list[str] | None:
    """Ensure the required API key env var is forwarded for ACP agent types.

    For non-ACP agent types (e.g. ``"default"``), *forward_env* is returned
    unchanged.  Raises ``ValueError`` if the required API key is not set in
    the current environment.
    """
    env_var = _ACP_ENV_VARS.get(agent_type)
    if env_var is None:
        return forward_env

    forward_env = list(forward_env or [])
    if env_var not in forward_env:
        if not os.getenv(env_var):
            raise ValueError(f"{env_var} not found in environment")
        forward_env.append(env_var)
    return forward_env


def setup_acp_workspace(agent_type: str, workspace: RemoteWorkspace) -> None:
    """Configure the workspace for ACP agents.

    For ``acp-claude``, writes ``~/.claude/settings.json`` to allow tool use
    without interactive permission prompts.
    """
    if agent_type != "acp-claude":
        return

    settings = {"permissions": {"allow": ["Edit", "Read", "Bash"]}}

    workspace.execute_command("mkdir -p ~/.claude")

    # Write via file_upload to avoid shell injection risks.
    fd, tmp_path = tempfile.mkstemp(suffix=".json", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(settings, f)
        result = workspace.file_upload(tmp_path, "~/.claude/settings.json")
        if not result.success:
            raise RuntimeError(
                f"Failed to upload Claude settings: {result}"
            )
        logger.info("Wrote Claude ACP settings to ~/.claude/settings.json")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
