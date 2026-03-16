"""Tests for ACP (Agent Communication Protocol) utilities."""

import os
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.utils.acp import (
    get_acp_command,
    get_acp_forward_env,
    is_acp_agent,
    setup_acp_workspace,
)
from benchmarks.utils.models import EvalMetadata, EvalOutput


# ---- is_acp_agent -----------------------------------------------------------


def test_is_acp_agent_claude():
    assert is_acp_agent("acp-claude") is True


def test_is_acp_agent_codex():
    assert is_acp_agent("acp-codex") is True


def test_is_acp_agent_default():
    assert is_acp_agent("default") is False


def test_is_acp_agent_unknown():
    assert is_acp_agent("something-else") is False


# ---- get_acp_command ---------------------------------------------------------


def test_get_acp_command_claude():
    assert get_acp_command("acp-claude") == ["claude-agent-acp"]


def test_get_acp_command_codex():
    assert get_acp_command("acp-codex") == ["codex-acp"]


def test_get_acp_command_unknown_raises():
    with pytest.raises(ValueError, match="Unknown ACP agent type"):
        get_acp_command("acp-unknown")


def test_get_acp_command_returns_copy():
    """Mutating the returned list should not affect future calls."""
    cmd = get_acp_command("acp-claude")
    cmd.append("--extra")
    assert get_acp_command("acp-claude") == ["claude-agent-acp"]


# ---- get_acp_forward_env ----------------------------------------------------


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"})
def test_forward_env_claude_appends_key_and_base_url():
    result = get_acp_forward_env("acp-claude", [])
    assert result is not None
    assert "ANTHROPIC_API_KEY" in result
    assert "ANTHROPIC_BASE_URL" in result


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
def test_forward_env_codex_appends_key_and_base_url():
    result = get_acp_forward_env("acp-codex", [])
    assert result is not None
    assert "OPENAI_API_KEY" in result
    assert "OPENAI_BASE_URL" in result


def test_forward_env_default_returns_unchanged():
    original = ["FOO"]
    result = get_acp_forward_env("default", original)
    assert result is original


def test_forward_env_default_none_returns_none():
    assert get_acp_forward_env("default") is None


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"})
def test_forward_env_none_becomes_list():
    result = get_acp_forward_env("acp-claude", None)
    assert result is not None
    assert "ANTHROPIC_API_KEY" in result
    assert "ANTHROPIC_BASE_URL" in result


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"})
def test_forward_env_does_not_duplicate():
    result = get_acp_forward_env(
        "acp-claude", ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"]
    )
    assert result is not None
    # Should not add duplicates
    assert result.count("ANTHROPIC_API_KEY") == 1
    assert result.count("ANTHROPIC_BASE_URL") == 1


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"})
def test_forward_env_preserves_existing():
    result = get_acp_forward_env("acp-claude", ["OTHER_VAR"])
    assert result is not None
    assert "OTHER_VAR" in result
    assert "ANTHROPIC_API_KEY" in result
    assert "ANTHROPIC_BASE_URL" in result


@patch.dict(os.environ, {}, clear=True)
def test_forward_env_missing_key_raises():
    # Remove the key entirely so getenv returns None
    os.environ.pop("ANTHROPIC_API_KEY", None)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not found"):
        get_acp_forward_env("acp-claude", [])


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"})
def test_forward_env_does_not_mutate_input():
    original = ["FOO"]
    result = get_acp_forward_env("acp-claude", original)
    assert original == ["FOO"]  # not mutated
    assert result is not None
    assert "FOO" in result
    assert "ANTHROPIC_API_KEY" in result
    assert "ANTHROPIC_BASE_URL" in result


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"})
def test_forward_env_accepts_tuple():
    """Tuples should be handled without error (converted to list)."""
    result = get_acp_forward_env("acp-claude", list(("EXISTING",)))
    assert result is not None
    assert "ANTHROPIC_API_KEY" in result
    assert "EXISTING" in result


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


def test_setup_acp_workspace_claude_uploads_settings():
    workspace = MagicMock()
    workspace.execute_command.return_value = MagicMock(exit_code=0)

    setup_acp_workspace("acp-claude", workspace)

    workspace.execute_command.assert_called_once()
    cmd = workspace.execute_command.call_args[0][0]
    assert "mkdir -p ~/.claude" in cmd
    assert "base64 -d" in cmd
    assert "settings.json" in cmd


# ---- metadata serialization -------------------------------------------------


def test_eval_metadata_omits_default_agent_type_from_serialization():
    metadata = EvalMetadata.model_construct(agent_type="default")

    assert "agent_type" not in metadata.model_dump()
    assert '"agent_type"' not in metadata.model_dump_json()


def test_eval_metadata_includes_non_default_agent_type_in_serialization():
    metadata = EvalMetadata.model_construct(agent_type="acp-claude")

    assert metadata.model_dump()["agent_type"] == "acp-claude"
    assert '"agent_type":"acp-claude"' in metadata.model_dump_json()


def test_eval_output_omits_default_agent_type_from_nested_metadata():
    output = EvalOutput(
        instance_id="instance-1",
        test_result={},
        metadata=EvalMetadata.model_construct(agent_type="default"),
    )

    dumped = output.model_dump()
    assert dumped["metadata"] is not None
    assert "agent_type" not in dumped["metadata"]
