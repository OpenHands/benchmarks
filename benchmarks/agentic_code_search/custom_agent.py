import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from openhands.sdk import Agent
from openhands.sdk.conversation import (
    ConversationState,
)
from openhands.sdk.logger import get_logger
from openhands.sdk.mcp import create_mcp_tools
from openhands.sdk.observability.laminar import (
    maybe_init_laminar,
)
from openhands.sdk.tool import ToolDefinition, resolve_tool


if TYPE_CHECKING:
    from openhands.sdk.conversation import ConversationState
logger = get_logger(__name__)
maybe_init_laminar()


class CustomAgent(Agent):
    def _initialize(self, state: "ConversationState"):
        """Create an AgentBase instance from an AgentSpec."""

        if self._tools:
            logger.warning("Agent already initialized; skipping re-initialization.")
            return

        tools: list[ToolDefinition] = []

        # Use ThreadPoolExecutor to parallelize tool resolution
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            # Submit tool resolution tasks
            for tool_spec in self.tools:
                future = executor.submit(resolve_tool, tool_spec, state)
                futures.append(future)

            # Submit MCP tools creation if configured
            if self.mcp_config:
                future = executor.submit(create_mcp_tools, self.mcp_config, 30)
                futures.append(future)

            # Collect results as they complete
            for future in futures:
                result = future.result()
                tools.extend(result)

        logger.info(
            f"Loaded {len(tools)} tools from spec: {[tool.name for tool in tools]}"
        )
        if self.filter_tools_regex:
            pattern = re.compile(self.filter_tools_regex)
            tools = [tool for tool in tools if pattern.match(tool.name)]
            logger.info(
                f"Filtered to {len(tools)} tools after applying regex filter: "
                f"{[tool.name for tool in tools]}",
            )

        # Do not include built-in tools; not subject to filtering
        # Instantiate built-in tools using their .create() method
        # for tool_class in BUILT_IN_TOOLS:
        #     tools.extend(tool_class.create(state))

        # Check tool types
        for tool in tools:
            if not isinstance(tool, ToolDefinition):
                raise ValueError(
                    f"Tool {tool} is not an instance of 'ToolDefinition'. "
                    f"Got type: {type(tool)}"
                )

        # Check name duplicates
        tool_names = [tool.name for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            duplicates = set(name for name in tool_names if tool_names.count(name) > 1)
            raise ValueError(f"Duplicate tool names found: {duplicates}")

        # Store tools in a dict for easy access
        self._tools = {tool.name: tool for tool in tools}
