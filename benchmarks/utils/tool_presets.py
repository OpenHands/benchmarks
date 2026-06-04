"""Shared tool-preset selection for benchmark runners."""

from benchmarks.utils.models import ToolPresetType
from openhands.sdk import Tool


def get_tools_for_preset(
    preset: ToolPresetType, enable_browser: bool = False
) -> list[Tool]:
    """Return tools for a benchmark tool preset."""
    if preset == "gemini":
        from openhands.tools.preset.gemini import get_gemini_tools

        return get_gemini_tools(enable_browser=enable_browser)
    if preset == "gpt5":
        from openhands.tools.preset.gpt5 import get_gpt5_tools

        return get_gpt5_tools(enable_browser=enable_browser)
    if preset == "planning":
        from openhands.tools.preset.planning import get_planning_tools

        # Planning preset does not support browser tools.
        return get_planning_tools()

    from openhands.tools.preset.default import get_default_tools

    return get_default_tools(enable_browser=enable_browser)
