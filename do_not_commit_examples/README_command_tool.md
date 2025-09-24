# Command Execution Tool for Runtime

This document explains the new command execution tool that has been added to the runtime environment.

## Overview

The command execution tool allows agents to run shell commands within the workspace environment during runtime execution. This is particularly useful for:

- Running tests and build commands
- Installing dependencies
- Executing scripts
- File system operations
- Any command-line operations needed for tasks

## Implementation

The tool consists of:

1. **`ExecuteCommandAction`**: Defines the command to execute with optional working directory and timeout
2. **`ExecuteCommandObservation`**: Contains the results including exit code, stdout, stderr
3. **`CommandExecutor`**: Handles the actual command execution using BashExecutor
4. **`create_command_tool()`**: Factory function to create the tool instance

## Integration

The tool is integrated into the runtime by:

1. **Import**: Added to `benchmarks/utils/run_evaluation.py`
2. **Registration**: Registered as "CommandTool" 
3. **Tool Specification**: Added to the tools list with workspace_root parameter
4. **Agent Creation**: Agent is created with both FileEditorTool and CommandTool

## Usage in Runtime

When `swe_bench/run_infer.py` is executed, the runtime will now have access to:

- **FileEditorTool**: For reading, writing, and editing files
- **CommandTool**: For executing shell commands

The agent can use these tools to:
- Examine the codebase
- Run tests to understand failures
- Install dependencies
- Execute build commands
- Run scripts and utilities
- Perform any command-line operations

## Workspace Management

The command tool integrates with the runtime's workspace management:

1. **Single Workspace**: The runtime creates a unique temporary workspace `/tmp/<random_string>/<repo_name>` for each instance
2. **Direct Access**: The agent works directly in this workspace - no copying needed
3. **Tool Integration**: CommandTool uses the workspace as the default working directory
4. **Concurrency Safe**: Each instance gets a unique workspace, preventing conflicts when running multiple instances of the same repository
5. **Simple Cleanup**: Only the original workspace needs cleanup - no synchronization required

## Security

The tool includes security measures:
- Commands are executed within the workspace root directory
- Working directory is validated to be within the workspace
- Uses the existing BashExecutor for consistent execution

## Example Usage

```python
# The agent can now use commands like:
execute_command(command="ls -la")
execute_command(command="python -m pytest tests/")
execute_command(command="pip install -r requirements.txt")
execute_command(command="git status")
```

## Testing

A test example is provided in `24_runtime_command_tool_example.py` that demonstrates:
- Tool registration and setup
- Command execution capabilities
- Integration with FileEditorTool
- Workspace-scoped operations

## Files Modified

1. **`benchmarks/utils/command_tool.py`** - New tool implementation
2. **`benchmarks/utils/run_evaluation.py`** - Integration into runtime
3. **`do_not_commit_examples/24_runtime_command_tool_example.py`** - Test example
4. **`do_not_commit_examples/README_command_tool.md`** - This documentation

The runtime now provides comprehensive command execution capabilities for agents working on SWE-Bench and similar tasks.