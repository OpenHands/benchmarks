"""Test for OpenAgentSafety 422 error fix."""

from pydantic import SecretStr

from benchmarks.openagentsafety.run_infer import ServerCompatibleAgent
from openhands.sdk import LLM, Tool


def test_server_compatible_agent_removes_forbidden_llm_fields():
    """Test that ServerCompatibleAgent.model_dump() excludes forbidden LLM fields."""
    # Create an LLM with forbidden fields
    llm = LLM(
        model="test-model",
        api_key=SecretStr("test-key"),
        extra_headers={"X-Custom": "value"},
        reasoning_summary="detailed",
        litellm_extra_body={"custom": "data"},
        temperature=0.7,
    )

    # Create agent with this LLM
    tools = [Tool(name="BashTool", params={})]
    agent = ServerCompatibleAgent(llm=llm, tools=tools)

    # Serialize the agent as would be sent to server
    agent_data = agent.model_dump()

    # Verify forbidden LLM fields are excluded
    assert "extra_headers" not in agent_data["llm"]
    assert "reasoning_summary" not in agent_data["llm"]
    assert "litellm_extra_body" not in agent_data["llm"]

    # Verify other LLM fields are preserved
    assert agent_data["llm"]["model"] == "test-model"
    assert agent_data["llm"]["temperature"] == 0.7

    # Verify the kind field is set to "Agent" for server compatibility
    assert agent_data["kind"] == "Agent"


def test_server_compatible_agent_with_minimal_llm():
    """Test that the agent works with an LLM without forbidden fields."""
    # Create a minimal LLM
    llm = LLM(
        model="test-model",
        temperature=0.5,
    )

    # Create agent
    tools = [Tool(name="BashTool", params={})]
    agent = ServerCompatibleAgent(llm=llm, tools=tools)

    # Verify it serializes without errors
    agent_data = agent.model_dump()
    assert agent_data["llm"]["model"] == "test-model"
    assert agent_data["llm"]["temperature"] == 0.5
    assert agent_data["kind"] == "Agent"


def test_server_compatible_agent_preserves_tools():
    """Test that tools are properly preserved in serialization."""
    # Create agent with multiple tools
    llm = LLM(model="test-model")
    tools = [
        Tool(name="BashTool", params={}),
        Tool(name="FileEditorTool", params={}),
        Tool(name="TaskTrackerTool", params={}),
    ]
    agent = ServerCompatibleAgent(llm=llm, tools=tools)

    # Serialize and verify tools are preserved
    agent_data = agent.model_dump()
    assert len(agent_data["tools"]) == 3
    assert agent_data["tools"][0]["name"] == "BashTool"
    assert agent_data["tools"][1]["name"] == "FileEditorTool"
    assert agent_data["tools"][2]["name"] == "TaskTrackerTool"
