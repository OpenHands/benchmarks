"""
Custom command execution tool for runtime environments.

This tool provides command execution capabilities within the runtime workspace,
allowing agents to run commands during instance processing.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from pydantic import Field

from openhands.sdk import ImageContent, TextContent
from openhands.sdk.tool import ActionBase, ObservationBase, ToolExecutor
from openhands.tools.execute_bash import BashExecutor, ExecuteBashAction


class ExecuteCommandAction(ActionBase):
    """Action to execute a command in the runtime workspace."""

    command: str = Field(description="Command to execute")
    working_dir: str | None = Field(
        default=None,
        description="Working directory for command execution (defaults to workspace root)",
    )
    timeout: int = Field(
        default=30, description="Timeout in seconds for command execution"
    )


class ExecuteCommandObservation(ObservationBase):
    """Observation containing command execution results."""

    command: str = Field(description="The command that was executed")
    exit_code: int = Field(description="Exit code of the command")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    timeout_occurred: bool = Field(
        default=False, description="Whether command timed out"
    )

    @property
    def agent_observation(self) -> Sequence[TextContent | ImageContent]:
        """Format observation for agent consumption."""
        if self.timeout_occurred:
            result = (
                f"Command timed out after {self.timeout} seconds:\n{self.command}\n"
            )
        else:
            result = f"Command executed (exit code {self.exit_code}):\n{self.command}\n"

        if self.stdout:
            result += f"\nSTDOUT:\n{self.stdout}"

        if self.stderr:
            result += f"\nSTDERR:\n{self.stderr}"

        return [TextContent(text=result)]


class CommandExecutor(ToolExecutor[ExecuteCommandAction, ExecuteCommandObservation]):
    """Executor for command execution in runtime workspace."""

    def __init__(self, workspace_root: str):
        """Initialize with workspace root directory."""
        self.workspace_root = os.path.abspath(workspace_root)

    def __call__(self, action: ExecuteCommandAction) -> ExecuteCommandObservation:
        """Execute the command and return observation."""
        working_dir = action.working_dir or self.workspace_root

        # Ensure working directory is within workspace for security
        abs_working_dir = os.path.abspath(working_dir)
        if not abs_working_dir.startswith(self.workspace_root):
            return ExecuteCommandObservation(
                command=action.command,
                exit_code=1,
                stderr=f"Working directory {working_dir} is outside workspace {self.workspace_root}",
                timeout_occurred=False,
            )

        # Use BashExecutor for consistent execution with correct working directory
        bash_executor = BashExecutor(working_dir=working_dir)
        bash_action = ExecuteBashAction(command=action.command)

        try:
            result = bash_executor(bash_action)
            return ExecuteCommandObservation(
                command=action.command,
                exit_code=result.exit_code,
                stdout=result.output,
                stderr="",  # BashExecutor combines stdout/stderr
                timeout_occurred=False,
            )
        except Exception as e:
            return ExecuteCommandObservation(
                command=action.command,
                exit_code=1,
                stderr=f"Execution error: {str(e)}",
                timeout_occurred=False,
            )


def create_command_tool(workspace_root: str = "/workspace"):
    """Create a command execution tool for the given workspace."""
    from openhands.sdk import Tool

    executor = CommandExecutor(workspace_root)

    return Tool(
        name="execute_command",
        description="""Execute commands in the runtime workspace.
        
This tool allows you to run shell commands within the workspace environment.
Use this for:
- Running tests and build commands
- Installing dependencies
- Executing scripts
- File system operations
- Any command-line operations needed for the task

The command will be executed in the workspace root directory unless a different
working_dir is specified. All paths should be relative to the workspace root.

Examples:
- execute_command(command="ls -la")
- execute_command(command="python -m pytest tests/", working_dir=".")
- execute_command(command="pip install -r requirements.txt")
""",
        action_type=ExecuteCommandAction,
        observation_type=ExecuteCommandObservation,
        executor=executor,
    )
