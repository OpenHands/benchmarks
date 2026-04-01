"""Utilities for ACP (Agent Communication Protocol) agent support."""

import base64
import json
import os
import threading
from contextlib import contextmanager
from typing import Any, cast

from benchmarks.utils.laminar import LMNR_ENV_VARS
from openhands.sdk import get_logger
from openhands.sdk.agent import ACPAgent
from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)

# Default timeout for ACP prompt() calls in seconds.
# Claude Opus 4.6 with extended thinking can take >30 min for complex tasks,
# so we use 60 min (3600s) as the default to avoid premature timeouts.
ACP_PROMPT_TIMEOUT: float = 3600.0

# Mapping of ACP agent types to the env vars they require.
# Both the API key and base URL are needed to route through LiteLLM proxy.
_ACP_ENV_VARS: dict[str, list[str]] = {
    "acp-claude": ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"],
    "acp-codex": ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
    "acp-gemini": ["GEMINI_API_KEY", "GEMINI_BASE_URL"],
}

# Mapping of ACP agent types to their ACP command.
_ACP_COMMANDS: dict[str, list[str]] = {
    "acp-claude": ["claude-agent-acp"],
    "acp-codex": ["codex-acp"],
    "acp-gemini": ["gemini", "--acp"],
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
    """Ensure the required env vars are forwarded for ACP agent types.

    For non-ACP agent types (e.g. ``"default"``), *forward_env* is returned
    unchanged.

    For ACP agent types, LMNR_ENV_VARS are included in forward_env to enable
    Laminar tracing within the workspace.  Provider credentials (API keys,
    base URLs) are **not** forwarded here — they are passed via
    ``ACPAgent.acp_env`` in :func:`build_acp_agent` to avoid leaking them
    into logged workspace payloads.
    """
    if agent_type not in _ACP_ENV_VARS:
        return forward_env

    forward_env = list(forward_env or [])

    # Include Laminar env vars for tracing in ACP agents
    for lmnr_var in LMNR_ENV_VARS:
        if lmnr_var not in forward_env:
            forward_env.append(lmnr_var)

    return forward_env


def extract_acp_model_hint(llm_model: str) -> str | None:
    """Extract a bare model identifier from a LiteLLM proxy model string.

    LLM configs use LiteLLM proxy paths like 'litellm_proxy/anthropic/claude-opus-4-6'.
    ACP servers need the bare model identifier (e.g. 'claude-opus-4-6') to match
    against their available models list.

    Strips the ``litellm_proxy/`` prefix and any provider segment (e.g. ``anthropic/``).
    Returns None for empty model strings.  Does not filter by provider — the
    ACP server is responsible for rejecting unsupported models.
    """
    if not llm_model:
        return None
    # Strip litellm_proxy/ prefix
    model = llm_model
    if model.startswith("litellm_proxy/"):
        model = model[len("litellm_proxy/") :]
    # Strip provider prefix (e.g., anthropic/)
    if "/" in model:
        model = model.rsplit("/", 1)[-1]
    return model


def _get_acp_env(agent_type: str) -> dict[str, str]:
    """Build the ``acp_env`` dict for the given ACP *agent_type*.

    Reads the provider credentials from the current process environment and
    returns them as a dict suitable for ``ACPAgent.acp_env``.  This keeps
    credentials out of the workspace ``forward_env`` (which is logged) and
    instead passes them directly to the ACP subprocess.

    Raises ``ValueError`` if the required API key is not set.
    """
    env_var_names = _ACP_ENV_VARS.get(agent_type, [])
    acp_env: dict[str, str] = {}
    for var in env_var_names:
        value = os.getenv(var)
        if value:
            acp_env[var] = value
        elif "API_KEY" in var:
            raise ValueError(f"{var} not found in environment")
    return acp_env


def build_acp_agent(agent_type: str, llm_model: str) -> ACPAgent:
    """Create an ACPAgent while remaining compatible with older type stubs.

    Provider credentials are passed via ``acp_env`` so they reach the ACP
    subprocess without being included in the workspace ``forward_env``.
    """
    return cast(Any, ACPAgent)(
        acp_command=get_acp_command(agent_type),
        acp_model=extract_acp_model_hint(llm_model),
        acp_prompt_timeout=ACP_PROMPT_TIMEOUT,
        acp_env=_get_acp_env(agent_type),
    )


def add_acp_agent_metadata(
    test_result: dict[str, Any],
    conversation: Any,
) -> None:
    """Add ACP agent metadata to an eval result payload.

    The ACPAgent stores agent_name/version in ``state.agent_state`` during init.
    This syncs back in the ``full_state`` ConversationStateUpdateEvent.  We scan
    the conversation events (last to first) to extract it.
    """
    name = ""
    version = ""
    for ev in reversed(list(conversation.state.events)):
        ev_dict = ev if isinstance(ev, dict) else getattr(ev, "__dict__", {})
        if ev_dict.get("key") == "full_state":
            val = ev_dict.get("value", {})
            agent_state = val.get("agent_state", {}) if isinstance(val, dict) else {}
            name = agent_state.get("acp_agent_name", "")
            version = agent_state.get("acp_agent_version", "")
            if name:
                break
    test_result["acp_agent_name"] = name
    test_result["acp_agent_version"] = version


def setup_acp_workspace(agent_type: str, workspace: RemoteWorkspace) -> None:
    """Configure the workspace for ACP agents.

    For ``acp-claude``, writes ``~/.claude/settings.json`` to allow tool use
    without interactive permission prompts.
    """
    if agent_type != "acp-claude":
        return

    settings = {"permissions": {"allow": ["Edit", "Read", "Bash"]}}
    settings_json = json.dumps(settings)

    # Use execute_command with base64 encoding to safely write the file,
    # avoiding both shell injection and file_upload issues with tilde expansion.
    encoded = base64.b64encode(settings_json.encode()).decode()
    result = workspace.execute_command(
        f"mkdir -p ~/.claude && echo '{encoded}' | base64 -d > ~/.claude/settings.json"
    )
    if result.exit_code != 0:
        raise RuntimeError(f"Failed to write Claude settings: {result.stderr}")
    logger.info("Wrote Claude ACP settings to ~/.claude/settings.json")


@contextmanager
def workspace_keepalive(
    agent_type: str, workspace: RemoteWorkspace, interval: int = 60
):
    """Keep the runtime workspace alive during ACP agent execution.

    ACP agents (Claude Code, Codex) use their own built-in tools and do not
    make calls to the workspace while thinking.  Without periodic activity the
    runtime management system considers the workspace idle and terminates it
    (default idle timeout ~20 min).

    This context manager spawns a daemon thread that periodically runs a no-op
    command (``true``) on the workspace to prevent idle termination.

    For non-ACP agent types this is a no-op pass-through.

    Important: Sends an immediate ping on context entry to reset the idle timer,
    then continues pinging at the specified interval.
    """
    if not is_acp_agent(agent_type):
        yield
        return

    stop = threading.Event()

    def _ping() -> None:
        # Ping immediately on thread start, then at regular intervals
        while True:
            try:
                workspace.execute_command("true")
                logger.debug("Workspace keep-alive ping sent")
            except Exception:
                logger.debug("Workspace keep-alive ping failed", exc_info=True)
            if stop.wait(interval):
                break

    t = threading.Thread(target=_ping, daemon=True)
    t.start()
    logger.info("Started workspace keep-alive (interval=%ds)", interval)
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=5)
        logger.info("Stopped workspace keep-alive")
