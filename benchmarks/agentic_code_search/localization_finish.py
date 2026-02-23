"""Custom finish tool for code localization tasks.

This tool allows the agent to submit localization results in a structured format where:
- File path is required
- Class name is optional
- Function name is optional
"""

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from rich.text import Text

from openhands.sdk import Action, Observation, ToolDefinition
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.tool import ToolAnnotations, ToolExecutor


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation


class CodeLocation(BaseModel):
    """A single code location with optional class and function."""

    file: str = Field(description="Path to the file (required)")
    class_name: str | None = Field(default=None, description="Class name (optional)")
    function_name: str | None = Field(
        default=None, description="Function/method name (optional)"
    )


class LocalizationFinishAction(Action):
    """Action for submitting final localization results."""

    locations: list[CodeLocation] = Field(
        description="""List of code locations to modify. Each location in this list must have:
- file: Path to the file relative to the repository root (required)
- class_name: Class name (optional, omit for changes to imports, global variables, and global functions)
- function_name: Function/method name (optional, omit for changes that edit parts of a file outside of any particular function)
"""
    )

    # message: str = Field(description="Code localization submission sent to the user.")

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this action."""
        content = Text()
        content.append("Submitting localization results:\n", style="bold blue")
        content.append(f"Found {len(self.locations)} location(s):\n", style="green")
        for i, loc in enumerate(self.locations, 1):
            content.append(f"  {i}. {loc.file}", style="cyan")
            if loc.class_name:
                content.append(f" → {loc.class_name}", style="yellow")
            if loc.function_name:
                content.append(f".{loc.function_name}", style="magenta")
            content.append("\n")
        return content


class LocalizationFinishObservation(Observation):
    """Observation returned after submitting localization results. No observation is needed since the agent will exit after this action."""

    @property
    def visualize(self) -> Text:
        """Return an empty Text representation since the message is in the action."""
        return Text()


def locations_to_dict_list(locations: list[CodeLocation]) -> list[dict]:
    """Convert CodeLocation objects to dictionary format.

    Args:
        locations: List of CodeLocation objects

    Returns:
        List of dictionaries with 'file', 'class_name', 'function_name' keys
    """
    return [
        {
            "file": loc.file,
            "class_name": loc.class_name,
            "function_name": loc.function_name,
        }
        for loc in locations
    ]


class LocalizationFinishExecutor(ToolExecutor):
    def __call__(
        self,
        action: LocalizationFinishAction,
        conversation: "BaseConversation | None" = None,  # noqa: ARG002
    ) -> LocalizationFinishObservation:
        try:
            loc_dict = locations_to_dict_list(action.locations)
            text = json.dumps(loc_dict, indent=2)
            if isinstance(conversation, BaseConversation):
                conversation.state.execution_status = (
                    ConversationExecutionStatus.FINISHED
                )  # type:ignore
            return LocalizationFinishObservation.from_text(text=text)
        except Exception as _:
            return LocalizationFinishObservation.from_text(text="")


TOOL_DESCRIPTION = """Submit your final code localization results.

Use this tool when you have identified all relevant files, classes, and functions that need to be modified to address the issue described in the problem statement.

Provide a structured list of locations. Each location must have:
- file: Path to the file relative to the root of the repository (required)
- class_name: Class name (optional)
- function_name: Function/method name (optional)

You must submit a list of locations that require modification and for each location you must follow the below rules in your output:
1. If the required modifications belong to a specific function that belongs to a class, provide the file path, class name, and function name.
2. If the required modification belongs to a function that is not part of any class, provide the file path and function name.
3. If the required modification does not belong to any specific class or a function (e.g. global variables, imports, new class, new global function etc.), it is sufficient to provide only the file path.
4. If the required modification belongs to a class (e.g. adding a new method to a class, changing the class inheritance), provide the file path and class name. If you are modifying the __init__ method of a class, you should provide the function name as well.

IMPORTANT:
1. If multiple different edits need to be edited in the same file, you should create separate entries for each edit, specifying the same file path but different class/function names as applicable. Each entry should compulsorily include the file path.
2. Do NOT include duplicate entries in your output for which the file, class, and function names are all identical.
3. Ensure that the file paths are accurate and relative to the root of the repository without any leading "./" or "/". All locations must be valid and exist in the codebase and this applies to class and function names as well.
4. Aim for high precision (all returned locations are relevant) and high recall (no relevant locations missed).
5. The agent will terminate its execution after you call this tool.
"""


class LocalizationFinishTool(
    ToolDefinition[LocalizationFinishAction, LocalizationFinishObservation]
):
    """Tool for submitting final localization results."""

    """Tool for submitting final code localization results."""

    @classmethod
    def create(
        cls,
        conv_state,  # noqa: ARG003
        **params,
    ) -> Sequence["LocalizationFinishTool"]:
        """Create LocalizationFinishTool instance.

        Args:
            conv_state: Conversation state (provides workspace info)
            workspace_dir: Optional workspace directory override
            **params: Additional parameters

        Returns:
            A sequence containing a single LocalizationFinishTool instance.
        """
        if params:
            raise ValueError("LocalizationFinishTool doesn't accept parameters")

        return [
            cls(
                name="localization_finish",  # type: ignore
                action_type=LocalizationFinishAction,
                observation_type=LocalizationFinishObservation,
                description=TOOL_DESCRIPTION,
                executor=LocalizationFinishExecutor(),
                annotations=ToolAnnotations(
                    title="localization_finish",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


# @tool(name="localization_finish")
# def _make_localization_finish_tool() -> list[ToolDefinition]:
#     """Create localization finish tool.

#     This is a localization-specific finish tool that accepts structured locations
#     and validates the output format.
#     """
#     return LocalizationFinishTool.create()
