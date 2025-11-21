"""OpenHands Benchmarking Suite"""

# Pre-import these tools to register pydantic models
# for serialization/deserialization.
from openhands.tools.file_editor import FileEditorTool  # noqa: F401
from openhands.tools.task_tracker import TaskTrackerTool  # noqa: F401
from openhands.tools.terminal import TerminalTool  # noqa: F401
