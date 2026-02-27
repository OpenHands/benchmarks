"""Utilities for ACP (Agent Communication Protocol) agent support."""

# Mapping of ACP agent types to the API key env vars they require.
_ACP_ENV_VARS: dict[str, str] = {
    "acp-claude": "ANTHROPIC_API_KEY",
    "acp-codex": "OPENAI_API_KEY",
}


def get_acp_forward_env(
    agent_type: str, forward_env: list[str] | None = None
) -> list[str] | None:
    """Ensure the required API key env var is forwarded for ACP agent types.

    For non-ACP agent types (e.g. ``"default"``), *forward_env* is returned
    unchanged.
    """
    env_var = _ACP_ENV_VARS.get(agent_type)
    if env_var is None:
        return forward_env

    forward_env = list(forward_env or [])
    if env_var not in forward_env:
        forward_env.append(env_var)
    return forward_env
