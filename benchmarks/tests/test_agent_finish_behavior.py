"""Test agent finish_on_message_only behavior for GPT-5 codex issue #78."""

from openhands.sdk.agent import Agent
from openhands.sdk.llm import LLM


def test_default_finish_on_message_only():
    """Test that default behavior is finish_on_message_only=True."""
    # Create a minimal LLM instance for testing
    llm = LLM(model="test-model")
    agent = Agent(
        llm=llm,
        tools=[],  # No tools to avoid registration issues
    )

    # Default should be True for backward compatibility
    assert agent.finish_on_message_only is True


def test_finish_on_message_only_parameter():
    """Test that finish_on_message_only parameter can be set."""
    llm = LLM(model="test-model")

    agent_true = Agent(
        llm=llm,
        tools=[],
        finish_on_message_only=True,
    )
    assert agent_true.finish_on_message_only is True

    agent_false = Agent(
        llm=llm,
        tools=[],
        finish_on_message_only=False,
    )
    assert agent_false.finish_on_message_only is False


def test_swe_bench_configuration():
    """Test that SWE-bench style configuration works correctly."""
    # This simulates the configuration used in SWE-bench
    llm = LLM(model="test-model")
    agent = Agent(
        llm=llm,
        tools=[],  # No tools to avoid registration issues
        finish_on_message_only=False,  # SWE-bench sets this to False
    )

    # Should have the parameter set correctly
    assert agent.finish_on_message_only is False
