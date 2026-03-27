"""Tests for benchmarks.utils.tools module."""

from benchmarks.utils.tools import get_tools_for_preset
from openhands.sdk import Tool


def test_get_tools_for_preset_default():
    """Test that default preset returns the expected tools."""
    tools = get_tools_for_preset("default")
    assert isinstance(tools, list)
    assert len(tools) > 0
    assert all(isinstance(tool, Tool) for tool in tools)


def test_get_tools_for_preset_gemini():
    """Test that gemini preset returns the expected tools."""
    tools = get_tools_for_preset("gemini")
    assert isinstance(tools, list)
    assert len(tools) > 0
    assert all(isinstance(tool, Tool) for tool in tools)


def test_get_tools_for_preset_gpt5():
    """Test that gpt5 preset returns the expected tools."""
    tools = get_tools_for_preset("gpt5")
    assert isinstance(tools, list)
    assert len(tools) > 0
    assert all(isinstance(tool, Tool) for tool in tools)


def test_get_tools_for_preset_planning():
    """Test that planning preset returns the expected tools."""
    tools = get_tools_for_preset("planning")
    assert isinstance(tools, list)
    assert len(tools) > 0
    assert all(isinstance(tool, Tool) for tool in tools)


def test_get_tools_for_preset_with_browser():
    """Test that enable_browser parameter is accepted."""
    # Test with browser enabled - should not raise
    tools_with_browser = get_tools_for_preset("default", enable_browser=True)
    assert isinstance(tools_with_browser, list)
    assert len(tools_with_browser) > 0

    # Planning preset doesn't support browser tools, but should still work
    planning_tools = get_tools_for_preset("planning", enable_browser=True)
    assert isinstance(planning_tools, list)
    assert len(planning_tools) > 0


def test_get_tools_for_preset_consistency():
    """Test that calling the function multiple times returns consistent results."""
    tools1 = get_tools_for_preset("gpt5")
    tools2 = get_tools_for_preset("gpt5")

    # Should return same number of tools
    assert len(tools1) == len(tools2)

    # Should return same tool types
    tool_types1 = [type(tool) for tool in tools1]
    tool_types2 = [type(tool) for tool in tools2]
    assert tool_types1 == tool_types2
