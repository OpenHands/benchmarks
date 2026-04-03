"""Tests for ACP (Agent Communication Protocol) utilities."""

import os
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.utils.acp import (
    _ACP_PROMPT_TIMEOUT_OVERRIDES,
    ACP_PROMPT_TIMEOUT,
    _get_acp_env,
    build_acp_agent,
    get_acp_command,
    get_acp_forward_env,
    is_acp_agent,
    setup_acp_workspace,
)


# ---- is_acp_agent -----------------------------------------------------------


def test_is_acp_agent_claude():
    assert is_acp_agent("acp-claude") is True


def test_is_acp_agent_codex():
    assert is_acp_agent("acp-codex") is True


def test_is_acp_agent_gemini():
    assert is_acp_agent("acp-gemini") is True


def test_is_acp_agent_default():
    assert is_acp_agent("default") is False


def test_is_acp_agent_unknown():
    assert is_acp_agent("something-else") is False


# ---- get_acp_command ---------------------------------------------------------


def test_get_acp_command_claude():
    assert get_acp_command("acp-claude") == ["claude-agent-acp"]


def test_get_acp_command_codex():
    assert get_acp_command("acp-codex") == ["codex-acp"]


def test_get_acp_command_gemini():
    assert get_acp_command("acp-gemini") == ["gemini", "--acp"]


def test_get_acp_command_unknown_raises():
    with pytest.raises(ValueError, match="Unknown ACP agent type"):
        get_acp_command("acp-unknown")


def test_get_acp_command_returns_copy():
    """Mutating the returned list should not affect future calls."""
    cmd = get_acp_command("acp-claude")
    cmd.append("--extra")
    assert get_acp_command("acp-claude") == ["claude-agent-acp"]


# ---- get_acp_forward_env ----------------------------------------------------
# After the security fix (#386), provider credentials are no longer added to
# forward_env.  Only LMNR (Laminar) tracing vars are forwarded.


def test_forward_env_claude_no_provider_keys():
    """Provider keys must NOT appear in forward_env (leak risk)."""
    result = get_acp_forward_env("acp-claude", [])
    assert result is not None
    assert "ANTHROPIC_API_KEY" not in result
    assert "ANTHROPIC_BASE_URL" not in result


def test_forward_env_codex_no_provider_keys():
    """Provider keys must NOT appear in forward_env (leak risk)."""
    result = get_acp_forward_env("acp-codex", [])
    assert result is not None
    assert "OPENAI_API_KEY" not in result
    assert "OPENAI_BASE_URL" not in result


def test_forward_env_gemini_no_provider_keys():
    """Provider keys must NOT appear in forward_env (leak risk)."""
    result = get_acp_forward_env("acp-gemini", [])
    assert result is not None
    assert "GEMINI_API_KEY" not in result
    assert "GEMINI_BASE_URL" not in result


def test_forward_env_default_returns_unchanged():
    original = ["FOO"]
    result = get_acp_forward_env("default", original)
    assert result is original


def test_forward_env_default_none_returns_none():
    assert get_acp_forward_env("default") is None


def test_forward_env_none_becomes_list():
    result = get_acp_forward_env("acp-claude", None)
    assert result is not None
    assert isinstance(result, list)


def test_forward_env_preserves_existing():
    result = get_acp_forward_env("acp-claude", ["OTHER_VAR"])
    assert result is not None
    assert "OTHER_VAR" in result
    # Provider keys must not leak into forward_env
    assert "ANTHROPIC_API_KEY" not in result


def test_forward_env_does_not_mutate_input():
    original = ["FOO"]
    result = get_acp_forward_env("acp-claude", original)
    assert original == ["FOO"]  # not mutated
    assert result is not None
    assert "FOO" in result


def test_forward_env_accepts_tuple():
    """Tuples should be handled without error (converted to list)."""
    result = get_acp_forward_env("acp-claude", list(("EXISTING",)))
    assert result is not None
    assert "EXISTING" in result


# ---- _get_acp_env -----------------------------------------------------------


@patch.dict(
    os.environ,
    {"ANTHROPIC_API_KEY": "sk-ant-test", "ANTHROPIC_BASE_URL": "https://proxy"},
)
def test_get_acp_env_claude():
    env = _get_acp_env("acp-claude")
    assert env == {
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "ANTHROPIC_BASE_URL": "https://proxy",
    }


@patch.dict(
    os.environ, {"OPENAI_API_KEY": "sk-oai-test", "OPENAI_BASE_URL": "https://proxy"}
)
def test_get_acp_env_codex():
    env = _get_acp_env("acp-codex")
    assert env == {"OPENAI_API_KEY": "sk-oai-test", "OPENAI_BASE_URL": "https://proxy"}


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=True)
def test_get_acp_env_omits_unset_base_url():
    """Base URL is optional — omitted when not set."""
    env = _get_acp_env("acp-claude")
    assert env == {"ANTHROPIC_API_KEY": "sk-test"}
    assert "ANTHROPIC_BASE_URL" not in env


@patch.dict(
    os.environ, {"GEMINI_API_KEY": "gem-test", "GEMINI_BASE_URL": "https://proxy"}
)
def test_get_acp_env_gemini():
    env = _get_acp_env("acp-gemini")
    assert env == {"GEMINI_API_KEY": "gem-test", "GEMINI_BASE_URL": "https://proxy"}


@patch.dict(os.environ, {}, clear=True)
def test_get_acp_env_missing_key_raises():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not found"):
        _get_acp_env("acp-claude")


def test_get_acp_env_default_returns_empty():
    assert _get_acp_env("default") == {}


# ---- build_acp_agent --------------------------------------------------------


@patch.dict(
    os.environ, {"ANTHROPIC_API_KEY": "sk-test", "ANTHROPIC_BASE_URL": "https://proxy"}
)
def test_build_acp_agent_passes_acp_env():
    """build_acp_agent should set acp_env with provider credentials."""
    agent = build_acp_agent("acp-claude", "litellm_proxy/anthropic/claude-opus-4-6")
    assert agent.acp_env == {
        "ANTHROPIC_API_KEY": "sk-test",
        "ANTHROPIC_BASE_URL": "https://proxy",
    }


@patch.dict(os.environ, {}, clear=True)
def test_build_acp_agent_missing_key_raises():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not found"):
        build_acp_agent("acp-claude", "litellm_proxy/anthropic/claude-opus-4-6")


@patch.dict(
    os.environ, {"GEMINI_API_KEY": "gk-test", "GEMINI_BASE_URL": "https://proxy"}
)
def test_build_acp_agent_gemini_uses_extended_timeout():
    """acp-gemini should use the longer prompt timeout from _ACP_PROMPT_TIMEOUT_OVERRIDES."""
    agent = build_acp_agent("acp-gemini", "litellm_proxy/gemini-3-flash-preview")
    assert agent.acp_prompt_timeout == _ACP_PROMPT_TIMEOUT_OVERRIDES["acp-gemini"]
    assert agent.acp_prompt_timeout > ACP_PROMPT_TIMEOUT


@patch.dict(
    os.environ, {"ANTHROPIC_API_KEY": "sk-test", "ANTHROPIC_BASE_URL": "https://proxy"}
)
def test_build_acp_agent_claude_uses_default_timeout():
    """acp-claude should use the default prompt timeout."""
    agent = build_acp_agent("acp-claude", "litellm_proxy/anthropic/claude-opus-4-6")
    assert agent.acp_prompt_timeout == ACP_PROMPT_TIMEOUT


# ---- setup_acp_workspace ----------------------------------------------------


def test_setup_acp_workspace_noop_for_default():
    workspace = MagicMock()
    setup_acp_workspace("default", workspace)
    workspace.execute_command.assert_not_called()
    workspace.file_upload.assert_not_called()


def test_setup_acp_workspace_noop_for_codex():
    workspace = MagicMock()
    setup_acp_workspace("acp-codex", workspace)
    workspace.execute_command.assert_not_called()
    workspace.file_upload.assert_not_called()


def test_setup_acp_workspace_noop_for_gemini():
    workspace = MagicMock()
    setup_acp_workspace("acp-gemini", workspace)
    workspace.execute_command.assert_not_called()
    workspace.file_upload.assert_not_called()


def test_setup_acp_workspace_claude_uploads_settings():
    workspace = MagicMock()
    workspace.execute_command.return_value = MagicMock(exit_code=0)

    setup_acp_workspace("acp-claude", workspace)

    workspace.execute_command.assert_called_once()
    cmd = workspace.execute_command.call_args[0][0]
    assert "mkdir -p ~/.claude" in cmd
    assert "base64 -d" in cmd
    assert "settings.json" in cmd
