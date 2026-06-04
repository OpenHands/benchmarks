"""Shared tool-preset selection for benchmark runners."""

from typing import assert_never

from benchmarks.utils.models import ToolPresetType
from openhands.sdk import Tool


def get_tools_for_preset(
    preset: ToolPresetType, enable_browser: bool = False
) -> list[Tool]:
    """Return tools for a benchmark tool preset."""
    match preset:
        case "gemini":
            from openhands.tools.preset.gemini import get_gemini_tools

            return get_gemini_tools(enable_browser=enable_browser)
        case "gpt5":
            from openhands.tools.preset.gpt5 import get_gpt5_tools

            return get_gpt5_tools(enable_browser=enable_browser)
        case "planning":
            from openhands.tools.preset.planning import get_planning_tools

            # Planning preset does not support browser tools.
            return get_planning_tools()
        case "default":
            from openhands.tools.preset.default import get_default_tools

            return get_default_tools(enable_browser=enable_browser)

    assert_never(preset)
