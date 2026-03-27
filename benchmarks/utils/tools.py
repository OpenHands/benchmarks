"""Shared utilities for tool preset management."""

from benchmarks.utils.models import ToolPresetType
from openhands.sdk import Tool


def get_tools_for_preset(
    preset: ToolPresetType, enable_browser: bool = False
) -> list[Tool]:
    """Get the list of tools for the given preset.

    Args:
        preset: The tool preset to use (default, gemini, gpt5, or planning).
        enable_browser: Whether to include browser tools.

    Returns:
        List of Tool classes for the given preset.
    """
    if preset == "gemini":
        from openhands.tools.preset.gemini import get_gemini_tools

        return get_gemini_tools(enable_browser=enable_browser)
    elif preset == "gpt5":
        from openhands.tools.preset.gpt5 import get_gpt5_tools

        return get_gpt5_tools(enable_browser=enable_browser)
    elif preset == "planning":
        from openhands.tools.preset.planning import get_planning_tools

        # Planning preset doesn't support browser tools
        return get_planning_tools()
    else:  # default
        from openhands.tools.preset.default import get_default_tools

        return get_default_tools(enable_browser=enable_browser)
