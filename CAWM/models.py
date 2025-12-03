import re
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

class ActionType(Enum):
    """Agent action type classification"""
    EXPLORATION = "exploration"      # search, grep, find, etc.
    FILE_VIEW = "file_view"          # view file content
    FILE_EDIT = "file_edit"          # edit file
    TESTING = "testing"              # run tests
    TERMINAL = "terminal"            # generic terminal command
    NAVIGATION = "navigation"        # cd, pwd, etc.
    VERSION_CONTROL = "version_control"  # git operations
    SETUP = "setup"                  # pip, conda, etc.
    THINK = "think"                  # thinking process
    OTHER = "other"                  # other actions

@dataclass
class WorkflowStep:
    """A single step in a workflow - compatible with llm_base.py"""
    env_description: str    # Environment state description before execution
    reasoning: str          # Why this action is taken
    action: str             # Abstracted action command
    action_type: str        # Action type string

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStep":
        return cls(
            env_description=data.get("env_description", ""),
            reasoning=data.get("reasoning", ""),
            action=data.get("action", ""),
            action_type=data.get("action_type", "other")
        )

@dataclass
class Workflow:
    """A reusable workflow"""
    id: str                              # Unique identifier
    description: str                     # Usage scenario description
    category: str                        # Category: exploration, investigation, modification, fix_and_verify, testing
    steps: List[WorkflowStep]            # List of steps
    level: int = 1                       # 1=General, 2=Specific
    source_instances: List[str] = field(default_factory=list)  # Source instance IDs
    frequency: int = 1                   # Occurrence frequency
    pattern: Tuple[str, ...] = field(default_factory=tuple)    # Action type pattern
    metadata: Dict[str, Any] = field(default_factory=dict)     # Extra metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "category": self.category,
            "steps": [s.to_dict() for s in self.steps],
            "level": self.level,
            "source_instances": self.source_instances,
            "frequency": self.frequency,
            "pattern": list(self.pattern),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        steps = [WorkflowStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            id=data["id"],
            description=data["description"],
            category=data.get("category", "general"),
            steps=steps,
            level=data.get("level", 1),
            source_instances=data.get("source_instances", []),
            frequency=data.get("frequency", 1),
            pattern=tuple(data.get("pattern", [])),
            metadata=data.get("metadata", {})
        )

# Forward declaration for type hinting
@dataclass
class TrajectoryEvent:
    """Parsed trajectory event"""
    index: int                           # Event index
    kind: str                            # Event type: ActionEvent, ObservationEvent
    action_type: ActionType              # Classified action type
    action: Dict[str, Any]               # Raw action data
    action_kind: str                     # Action kind: TerminalAction, FileEditorAction, ThinkAction
    thought: List[str]                   # Content of thinking
    command: Optional[str] = None        # Executed command (if any)
    path: Optional[str] = None           # Operated file path (if any)
    observation: Optional[Dict[str, Any]] = None  # Observation data
    observation_content: Optional[str] = None     # Observation text content
    tool_call_id: Optional[str] = None   # Tool call ID
    raw_event: Dict[str, Any] = field(default_factory=dict)  # Raw event data

    @classmethod
    def from_raw_event(cls, event: Dict[str, Any], index: int) -> Optional["TrajectoryEvent"]:
        """Parse from raw event"""
        kind = event.get("kind")
        # We mainly care about ActionEvent and ObservationEvent
        if kind not in ["ActionEvent", "ObservationEvent"]:
             return None

        action_data = event.get("action") or {}
        action_kind = action_data.get("kind", "")
        
        # Extract thoughts
        thoughts = []
        raw_thoughts = event.get("thought", [])
        if isinstance(raw_thoughts, str):
             thoughts.append(raw_thoughts)
        elif isinstance(raw_thoughts, list):
            for t in raw_thoughts:
                if isinstance(t, dict):
                    thoughts.append(t.get("text", ""))
                elif isinstance(t, str):
                    thoughts.append(t)

        command = action_data.get("command")
        path = action_data.get("path")
        
        # Note: action_type is set to OTHER initially; caller should set it using classify_action_type
        # or we can do it here if the function is defined.
        # To avoid circular dependency issues if any, we'll call the function here.
        act_type = classify_action_type(event)

        return cls(
            index=index,
            kind=kind,
            action_type=act_type,
            action=action_data,
            action_kind=action_kind,
            thought=thoughts,
            command=command,
            path=path,
            observation=event.get("observation"),
            observation_content=event.get("observation", {}).get("content") if event.get("observation") else None,
            tool_call_id=event.get("tool_call_id"),
            raw_event=event
        )

    def get_action_summary(self, max_length: int = 200) -> str:
        """Get action summary"""
        if self.action_kind == "TerminalAction":
            return f"[Terminal] {self.command}"[:max_length]
        elif self.action_kind == "FileEditorAction":
            cmd = self.action.get("command", "")
            return f"[{cmd}] {self.path}"[:max_length]
        elif self.action_kind == "ThinkAction" or self.action_type == ActionType.THINK:
            if self.thought and len(self.thought) > 0:
                return f"[Thought] {self.thought[0]}"[:max_length]
        return self.action_kind

    def is_key_step(self) -> bool:
        """Determine if it is a key step"""
        return self.action_type in [ActionType.FILE_EDIT, ActionType.TESTING]

@dataclass
class Trajectory:
    """Complete trajectory"""
    instance_id: str                     # Instance unique identifier
    instruction: str                     # Task instruction/problem description
    events: List[TrajectoryEvent]        # Parsed event list
    git_patch: Optional[str] = None      # Generated git patch
    repository: str = ""                 # Repository name (parsed from instance_id)
    issue_type: str = ""                 # Issue type (optional)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra metadata

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "Trajectory":
        """Parse from raw JSONL data"""
        instance_id = raw.get("instance_id", "unknown")
        repo = instance_id.split("__")[0] if "__" in instance_id else ""
        
        events = []
        history = raw.get("history", [])
        for i, event in enumerate(history):
            traj_event = TrajectoryEvent.from_raw_event(event, i)
            if traj_event:
                events.append(traj_event)
                
        return cls(
            instance_id=instance_id,
            instruction=raw.get("instruction", ""),
            events=events,
            git_patch=raw.get("test_result", {}).get("git_patch"),
            repository=repo,
            metadata={"test_result": raw.get("test_result")}
        )

    @classmethod
    def load_from_jsonl(cls, file_path: str) -> List["Trajectory"]:
        """Load multiple trajectories from JSONL file"""
        trajectories = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        trajectories.append(cls.from_raw(data))
                    except json.JSONDecodeError:
                        pass
        return trajectories

    def get_action_sequence(self) -> List[ActionType]:
        """Get action type sequence"""
        return [e.action_type for e in self.events]

    def get_key_events(self) -> List[TrajectoryEvent]:
        """Get key events"""
        return [e for e in self.events if e.is_key_step()]

    def __len__(self) -> int:
        return len(self.events)

@dataclass
class TrajectoryCluster:
    """Trajectory Cluster"""
    cluster_id: str                      # Cluster ID
    trajectories: List[Trajectory]       # List of trajectories in cluster
    label: str = ""                      # Cluster label/description
    similarity_method: str = ""          # Similarity method used
    centroid: Optional[Any] = None       # Cluster centroid (optional)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.trajectories)

    def get_instance_ids(self) -> List[str]:
        """Get all instance IDs"""
        return [t.instance_id for t in self.trajectories]

# Helper functions

def classify_action_type(event: Dict[str, Any]) -> ActionType:
    """Classify action type based on event content"""
    kind = event.get("kind")
    if kind != "ActionEvent":
        return ActionType.OTHER
        
    action = event.get("action") or {}
    action_kind = action.get("kind", "")
    command = action.get("command", "")
    
    if action_kind == "FileEditorAction":
        if command == "view":
            return ActionType.FILE_VIEW
        return ActionType.FILE_EDIT
        
    if action_kind == "TerminalAction":
        if not command:
             return ActionType.TERMINAL
        # Regex matching for command patterns
        if re.search(r'\b(find|grep|rg|ls|cat)\b', command):
            return ActionType.EXPLORATION
        if re.search(r'\b(pytest|python.*test|tox|npm test)\b', command):
            return ActionType.TESTING
        if re.search(r'\b(cd|pwd)\b', command):
            return ActionType.NAVIGATION
        if re.search(r'\b(git)\b', command):
            return ActionType.VERSION_CONTROL
        if re.search(r'\b(pip|conda|npm install)\b', command):
            return ActionType.SETUP
        return ActionType.TERMINAL
        
    if "think" in action_kind.lower():
        return ActionType.THINK
        
    return ActionType.OTHER

def abstract_path(path: str, repo_name: str = "") -> str:
    """Abstract concrete path to template"""
    if not path:
        return ""
    
    # 1. Try to replace standard workspace pattern first (most specific)
    if repo_name and f"/workspace/{repo_name}" in path:
        return path.replace(f"/workspace/{repo_name}", "{repo}")

    # 2. General heuristics for workspace
    if "/workspace/" in path:
         return re.sub(r'^.*?/workspace/[^/]+', '{repo}', path)

    # 3. Fallback: just replace repo name if visible
    if repo_name and repo_name in path:
        return path.replace(repo_name, "{repo}")
        
    return path

def abstract_command(command: str, repo_name: str = "") -> str:
    """Abstract concrete command to template"""
    if not command:
        return ""
    # Replace repo path
    cmd = abstract_path(command, repo_name)
    # More specific abstractions can be added here
    return cmd
