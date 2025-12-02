"""
AWM Integration for OpenHands SWE-bench Evaluation

This module integrates Agent Workflow Memory (AWM) with the OpenHands benchmarks
repository for SWE-bench evaluation. It provides:

1. Trajectory parsing from OpenHands event format
2. Workflow extraction using both rule-based and LLM-based methods
3. Integration with OpenHands Skills system
4. Online and offline AWM modes

Based on: "Agent Workflow Memory" (Wang et al., 2024)
Repository: https://github.com/OpenHands/benchmarks

Usage:
    # Offline mode - extract from training trajectories
    python awm_swebench_integration.py extract --input output.jsonl --output workflows.json

    # Apply workflows to new task
    python awm_swebench_integration.py apply --workflows workflows.json --task "Fix bug in X"
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
# Data Classes for Workflow Representation
# =============================================================================


@dataclass
class WorkflowStep:
    """
    A single step in a workflow, following AWM's format:
    - Environment description: What the agent observes
    - Reasoning: Why the agent takes this action
    - Action: The abstracted action taken
    """

    env_description: str
    reasoning: str
    action: str
    action_type: str  # exploration, file_view, file_edit, testing, navigation

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "WorkflowStep":
        return cls(**data)


@dataclass
class Workflow:
    """
    A reusable workflow extracted from bug-fixing trajectories.
    """

    id: str  # Unique identifier
    description: str  # Human-readable description
    category: str  # exploration, bugfix, testing, etc.
    steps: List[WorkflowStep]  # Sequence of steps
    source_instances: List[str] = field(default_factory=list)
    frequency: int = 1
    success_rate: float = 1.0
    avg_steps_saved: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "description": self.description,
            "category": self.category,
            "steps": [s.to_dict() for s in self.steps],
            "source_instances": self.source_instances,
            "frequency": self.frequency,
            "success_rate": self.success_rate,
            "avg_steps_saved": self.avg_steps_saved,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Workflow":
        steps = [WorkflowStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            id=data["id"],
            description=data["description"],
            category=data.get("category", "general"),
            steps=steps,
            source_instances=data.get("source_instances", []),
            frequency=data.get("frequency", 1),
            success_rate=data.get("success_rate", 1.0),
            avg_steps_saved=data.get("avg_steps_saved", 0.0),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# OpenHands Event Parsing
# =============================================================================


class OpenHandsEventParser:
    """
    Parse OpenHands event format into structured trajectories.

    Event types from uploaded files:
    - SystemPromptEvent: Initial system configuration
    - MessageEvent: User/agent messages
    - ActionEvent: Agent actions with thought, reasoning, action details
    - ObservationEvent: Results from actions
    - ConversationStateUpdateEvent: State changes
    """

    # Action type classification patterns
    EXPLORATION_PATTERNS = [
        r"^find\s",
        r"^grep\s",
        r"^ls\s",
        r"^cat\s",
        r"^head\s",
        r"^tail\s",
        r"^wc\s",
        r"^tree\s",
        r"^locate\s",
        r"^which\s",
    ]
    NAVIGATION_PATTERNS = [r"^cd\s", r"^pwd", r"^pushd", r"^popd"]
    TESTING_PATTERNS = [
        r"^pytest",
        r"^python.*test",
        r"^python -m pytest",
        r"^./test",
        r"^make test",
        r"^tox\s",
        r"^nose",
    ]
    SETUP_PATTERNS = [r"^pip\s", r"^conda\s", r"^apt", r"^npm\s", r"^yarn\s"]
    VERSION_CONTROL_PATTERNS = [r"^git\s", r"^diff\s", r"^patch\s"]

    def __init__(self):
        self.events = []

    def parse_event_file(self, filepath: Path) -> Dict:
        """Parse a single event JSON file."""
        with open(filepath) as f:
            return json.load(f)

    def parse_event_directory(self, dirpath: Path) -> List[Dict]:
        """Parse all event files from a directory."""
        events = []
        for f in sorted(dirpath.glob("event-*.json")):
            try:
                event = self.parse_event_file(f)
                events.append(event)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {f}")
        return events

    def parse_output_jsonl(self, filepath: Path) -> List[Dict]:
        """
        Parse output.jsonl file containing complete trajectories.

        Structure (from your uploaded file):
        {
            "instance_id": "astropy__astropy-12907",
            "test_result": {"git_patch": "..."},
            "instruction": "...",
            "history": [list of events],
            "metrics": {...}
        }
        """
        trajectories = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        traj = json.loads(line)
                        trajectories.append(traj)
                    except json.JSONDecodeError:
                        continue
        return trajectories

    def extract_action_observation_pairs(
        self, events: List[Dict]
    ) -> List[Tuple[Dict, Optional[Dict]]]:
        """
        Extract (ActionEvent, ObservationEvent) pairs.
        Each action should have a corresponding observation.
        """
        pairs = []
        i = 0
        while i < len(events):
            event = events[i]
            if event.get("kind") == "ActionEvent":
                # Look for corresponding ObservationEvent
                obs = None
                if i + 1 < len(events):
                    next_event = events[i + 1]
                    if next_event.get("kind") == "ObservationEvent":
                        # Match by tool_call_id
                        if next_event.get("tool_call_id") == event.get("tool_call_id"):
                            obs = next_event
                            i += 1
                pairs.append((event, obs))
            i += 1
        return pairs

    def classify_action_type(self, command: str) -> str:
        """Classify a terminal command into action types."""
        command = command.strip()

        for pattern in self.EXPLORATION_PATTERNS:
            if re.match(pattern, command):
                return "exploration"

        for pattern in self.NAVIGATION_PATTERNS:
            if re.match(pattern, command):
                return "navigation"

        for pattern in self.TESTING_PATTERNS:
            if re.match(pattern, command):
                return "testing"

        for pattern in self.SETUP_PATTERNS:
            if re.match(pattern, command):
                return "setup"

        for pattern in self.VERSION_CONTROL_PATTERNS:
            if re.match(pattern, command):
                return "version_control"

        # Check for python execution
        if re.match(r"^python", command):
            return "execution"

        return "terminal"

    def abstract_command(self, command: str) -> str:
        """
        Abstract a command by replacing specific values with placeholders.
        This is key to AWM's reusability.
        """
        # Replace specific paths
        command = re.sub(r"/workspace/[^\s/]+/", "{repo}/", command)
        command = re.sub(r"/home/[^\s/]+/", "{home}/", command)
        command = re.sub(r"/testbed/[^\s/]+/", "{repo}/", command)

        # Replace grep/find patterns with placeholders
        command = re.sub(
            r'grep -[a-zA-Z]+ "[^"]+"', 'grep {options} "{pattern}"', command
        )
        command = re.sub(
            r"grep -[a-zA-Z]+ '[^']+'", "grep {options} '{pattern}'", command
        )
        command = re.sub(r'grep "[^"]+"', 'grep "{pattern}"', command)

        # Replace file search patterns
        command = re.sub(r'-name "[^"]+"', '-name "{filename}"', command)
        command = re.sub(r"-name '[^']+'", "-name '{filename}'", command)

        # Replace specific Python files but keep structure
        command = re.sub(r"(\s|/)test_[a-z_]+\.py", r"\1{test_file}.py", command)

        # Replace specific line numbers in view ranges
        command = re.sub(r"\[\d+,\s*\d+\]", "[{start}, {end}]", command)
        command = re.sub(r"\[\d+,\s*-1\]", "[{start}, -1]", command)

        return command

    def abstract_file_path(self, path: str) -> str:
        """Abstract file paths for reusability."""
        # Replace repo-specific paths
        path = re.sub(r"/workspace/[^/]+/", "{repo}/", path)
        path = re.sub(r"/testbed/[^/]+/", "{repo}/", path)

        # Keep important directory names, abstract others
        parts = path.split("/")
        abstracted = []
        important_dirs = {"tests", "test", "src", "lib", "core", "utils", "models"}

        for part in parts:
            if part in important_dirs or part == "{repo}":
                abstracted.append(part)
            elif part.endswith(".py"):
                if part.startswith("test_"):
                    abstracted.append("{test_file}.py")
                else:
                    abstracted.append("{module}.py")
            elif part:
                abstracted.append("{dir}")

        return "/".join(abstracted)

    def parse_action_event(self, event: Dict) -> Optional[Dict]:
        """Parse an ActionEvent into structured format."""
        action_data = event.get("action", {})
        action_kind = action_data.get("kind", "")

        result = {
            "kind": action_kind,
            "raw_action": action_data,
            "thought": "",
            "reasoning": "",
            "abstracted_action": "",
            "action_type": "unknown",
        }

        # Extract thought (shown to user)
        thoughts = event.get("thought", [])
        if thoughts:
            for t in thoughts:
                if isinstance(t, dict) and t.get("type") == "text":
                    result["thought"] += t.get("text", "")

        # Extract reasoning (internal chain of thought)
        result["reasoning"] = event.get("reasoning_content", "") or ""

        # Parse based on action type
        if action_kind == "TerminalAction":
            command = action_data.get("command", "")
            result["abstracted_action"] = self.abstract_command(command)
            result["action_type"] = self.classify_action_type(command)
            result["raw_command"] = command

        elif action_kind == "FileEditorAction":
            cmd = action_data.get("command", "")
            path = action_data.get("path", "")
            abstracted_path = self.abstract_file_path(path)

            if cmd == "view":
                result["abstracted_action"] = f"view({abstracted_path})"
                result["action_type"] = "file_view"
            elif cmd == "create":
                result["abstracted_action"] = f"create({abstracted_path})"
                result["action_type"] = "file_create"
            elif cmd == "str_replace":
                result["abstracted_action"] = (
                    f"str_replace({abstracted_path}, {{old}}, {{new}})"
                )
                result["action_type"] = "file_edit"
            else:
                result["abstracted_action"] = f"{cmd}({abstracted_path})"
                result["action_type"] = "file_edit"

        return result

    def parse_observation_event(self, event: Dict) -> Optional[Dict]:
        """Parse an ObservationEvent into structured format."""
        obs_data = event.get("observation", {})

        result = {
            "kind": obs_data.get("kind", "unknown"),
            "is_error": obs_data.get("is_error", False),
            "exit_code": obs_data.get("metadata", {}).get("exit_code"),
            "summary": "",
        }

        # Extract and summarize content
        content = obs_data.get("content", [])
        if content:
            full_text = ""
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    full_text += c.get("text", "")
            result["summary"] = self._summarize_output(full_text)

        return result

    def _summarize_output(self, text: str, max_len: int = 200) -> str:
        """Create a brief summary of command output."""
        # Remove ANSI codes
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)

        # Detect common patterns
        lower = text.lower()
        if "error" in lower or "exception" in lower or "traceback" in lower:
            return "Error in output"
        if "test" in lower:
            if "passed" in lower or "ok" in lower:
                return "Tests passed"
            if "failed" in lower or "error" in lower:
                return "Tests failed"
        if "no such file" in lower or "not found" in lower:
            return "File/command not found"
        if text.strip() == "":
            return "Empty output"

        # Truncate
        lines = text.strip().split("\n")
        if len(lines) > 5:
            return f"Output: {len(lines)} lines ({len(text)} chars)"
        if len(text) > max_len:
            return text[:max_len] + "..."
        return text[:max_len]


# =============================================================================
# Workflow Extraction
# =============================================================================


class WorkflowExtractor:
    """
    Extract reusable workflows from trajectories.

    Implements both:
    1. Rule-based extraction (AWM_rule): Extract based on action patterns
    2. LLM-based extraction (AWM_lm): Use LLM to identify sub-routines
    """

    def __init__(self, use_llm: bool = False, llm_model: str = "gpt-4"):
        self.parser = OpenHandsEventParser()
        self.use_llm = use_llm
        self.llm_model = llm_model

    def trajectory_to_steps(
        self, events: List[Dict], instance_id: str
    ) -> List[WorkflowStep]:
        """Convert a trajectory into a sequence of WorkflowSteps."""
        pairs = self.parser.extract_action_observation_pairs(events)
        steps = []

        for action_event, obs_event in pairs:
            parsed_action = self.parser.parse_action_event(action_event)
            if not parsed_action or parsed_action["action_type"] == "unknown":
                continue

            # Skip navigation-only steps (cd, pwd)
            if parsed_action["action_type"] == "navigation":
                continue

            # Parse observation
            env_desc = ""
            if obs_event:
                parsed_obs = self.parser.parse_observation_event(obs_event)
                if parsed_obs:
                    env_desc = parsed_obs.get("summary", "")

            step = WorkflowStep(
                env_description=env_desc,
                reasoning=parsed_action.get("thought", "")[:500],  # Truncate
                action=parsed_action["abstracted_action"],
                action_type=parsed_action["action_type"],
            )
            steps.append(step)

        return steps

    def extract_ngram_patterns(
        self,
        all_trajectories: List[List[WorkflowStep]],
        min_length: int = 2,
        max_length: int = 8,
        min_frequency: int = 2,
    ) -> List[Tuple[Tuple[str, ...], List]]:
        """
        Extract common n-gram patterns based on action types.
        Returns list of (action_type_sequence, occurrences).
        """
        ngram_counts = defaultdict(list)

        for traj_idx, steps in enumerate(all_trajectories):
            action_types = tuple(s.action_type for s in steps)

            for length in range(min_length, min(len(action_types), max_length) + 1):
                for start in range(len(action_types) - length + 1):
                    ngram = action_types[start : start + length]
                    step_slice = steps[start : start + length]
                    ngram_counts[ngram].append((traj_idx, start, step_slice))

        # Filter by frequency
        patterns = [
            (ngram, occurrences)
            for ngram, occurrences in ngram_counts.items()
            if len(occurrences) >= min_frequency
        ]

        # Sort by frequency * length (prefer longer common patterns)
        patterns.sort(key=lambda x: len(x[1]) * len(x[0]), reverse=True)

        return patterns

    def deduplicate_workflows(self, workflows: List[Workflow]) -> List[Workflow]:
        """Remove redundant workflows (subsets of longer workflows)."""
        if not workflows:
            return []

        # Sort by length descending
        sorted_wfs = sorted(workflows, key=lambda w: len(w.steps), reverse=True)

        result = []
        seen_patterns: Set[Tuple[str, ...]] = set()

        for wf in sorted_wfs:
            pattern = tuple(s.action_type for s in wf.steps)

            # Check if this is a subset of an existing pattern
            is_subset = False
            for seen in seen_patterns:
                if len(pattern) < len(seen):
                    # Check if pattern is a contiguous subset
                    pattern_str = "-".join(pattern)
                    seen_str = "-".join(seen)
                    if pattern_str in seen_str:
                        is_subset = True
                        break

            if not is_subset:
                result.append(wf)
                seen_patterns.add(pattern)

        return result

    def generate_workflow_id(self, action_types: Tuple[str, ...]) -> str:
        """Generate a unique ID for a workflow based on its pattern."""
        pattern_str = "-".join(action_types)
        hash_val = hashlib.md5(pattern_str.encode()).hexdigest()[:8]
        return f"wf-{hash_val}"

    def generate_description(
        self, action_types: Tuple[str, ...], steps: List[WorkflowStep]
    ) -> str:
        """Generate human-readable description for a workflow."""
        # Common SWE-bench patterns
        pattern_map = {
            ("exploration",): "Search for relevant files in the codebase",
            ("exploration", "exploration"): "Comprehensive codebase search",
            ("exploration", "file_view"): "Find and examine source files",
            ("file_view",): "Examine source code",
            ("file_view", "file_view"): "Review multiple source files",
            ("file_view", "file_edit"): "View and modify source code",
            ("file_edit",): "Modify source code",
            ("file_edit", "testing"): "Apply fix and run tests",
            ("testing",): "Run tests to verify behavior",
            ("exploration", "file_view", "file_edit"): "Locate, examine, and fix bug",
            (
                "exploration",
                "file_view",
                "file_edit",
                "testing",
            ): "Complete bug-fix cycle: find, understand, fix, verify",
            (
                "file_view",
                "file_edit",
                "testing",
            ): "Review code, apply fix, verify with tests",
        }

        if action_types in pattern_map:
            return pattern_map[action_types]

        # Generate based on sequence
        action_names = {
            "exploration": "explore",
            "file_view": "view",
            "file_edit": "edit",
            "testing": "test",
            "setup": "setup",
            "terminal": "execute",
            "version_control": "check diff",
        }

        readable_actions = [action_names.get(a, a) for a in action_types]
        if len(readable_actions) <= 3:
            return " → ".join(readable_actions).capitalize() + " workflow"
        else:
            return f"{readable_actions[0]} to {readable_actions[-1]} workflow ({len(readable_actions)} steps)"

    def categorize_workflow(self, action_types: Tuple[str, ...]) -> str:
        """Categorize workflow based on its action composition."""
        if "testing" in action_types:
            if "file_edit" in action_types:
                return "fix_and_verify"
            return "testing"
        if "file_edit" in action_types:
            return "modification"
        if "file_view" in action_types:
            if "exploration" in action_types:
                return "investigation"
            return "review"
        if "exploration" in action_types:
            return "exploration"
        return "general"

    def extract_rule_based(
        self, trajectories: List[Dict], only_successful: bool = True
    ) -> List[Workflow]:
        """
        Rule-based workflow extraction (AWM_rule).

        Steps:
        1. Filter to successful trajectories
        2. Convert to WorkflowSteps
        3. Find common n-gram patterns
        4. Create Workflow objects
        5. Deduplicate
        """
        all_steps = []
        instance_ids = []

        for traj in trajectories:
            # Filter unsuccessful if requested
            if only_successful:
                test_result = traj.get("test_result", {})
                if not test_result.get("git_patch"):
                    continue

            instance_id = traj.get("instance_id", "unknown")
            events = traj.get("history", [])

            steps = self.trajectory_to_steps(events, instance_id)
            if steps:
                all_steps.append(steps)
                instance_ids.append(instance_id)

        if not all_steps:
            return []

        # For single trajectory, lower min_frequency
        min_freq = 1 if len(all_steps) < 3 else 2

        # Extract patterns
        patterns = self.extract_ngram_patterns(
            all_steps, min_length=2, max_length=8, min_frequency=min_freq
        )

        # Create workflows
        workflows = []
        for action_types, occurrences in patterns:
            # Use first occurrence as template
            _, _, template_steps = occurrences[0]

            # Collect source instances
            source_ids = list(
                set(
                    instance_ids[occ[0]]
                    for occ in occurrences
                    if occ[0] < len(instance_ids)
                )
            )

            wf = Workflow(
                id=self.generate_workflow_id(action_types),
                description=self.generate_description(action_types, template_steps),
                category=self.categorize_workflow(action_types),
                steps=template_steps,
                source_instances=source_ids,
                frequency=len(occurrences),
            )
            workflows.append(wf)

        # Deduplicate
        workflows = self.deduplicate_workflows(workflows)

        return workflows

    def extract_llm_based(self, trajectories: List[Dict]) -> List[Workflow]:
        """
        LLM-based workflow extraction (AWM_lm).

        Uses an LLM to identify and abstract common sub-routines.
        More accurate but requires API calls.
        """
        if not self.use_llm:
            print("LLM-based extraction disabled. Use extract_rule_based instead.")
            return []

        # Format trajectories for LLM
        # prompt = self._build_extraction_prompt(trajectories)

        # TODO: Implement actual LLM call
        # This would use the OpenHands LLM interface
        print("LLM extraction not yet implemented")
        return []

    def _build_extraction_prompt(self, trajectories: List[Dict]) -> str:
        """Build prompt for LLM-based extraction."""
        return """Given these bug-fixing trajectories from SWE-bench, extract common workflows.

Each trajectory shows an agent solving a GitHub issue. Extract reusable sub-routines that:
1. Appear across multiple trajectories
2. Have clear start and end points
3. Can be abstracted to apply to similar bugs

Format each workflow as:
## Workflow: [Description]
Category: [exploration|investigation|modification|fix_and_verify|testing]

Step 1:
  Reasoning: [Why this step]
  Action: [Abstracted action]

Step 2:
  ...

Trajectories:
{trajectories}
"""


# =============================================================================
# Workflow Memory Management
# =============================================================================


class SWEBenchWorkflowMemory:
    """
    Main class for AWM applied to SWE-bench.

    Provides:
    - Workflow storage and retrieval
    - Relevance-based workflow selection
    - Prompt formatting for agent integration
    - Online and offline operation modes
    """

    # Pre-defined workflows for SWE-bench
    BASE_WORKFLOWS = [
        Workflow(
            id="wf-explore-repo",
            description="Explore repository structure to understand codebase",
            category="exploration",
            steps=[
                WorkflowStep(
                    env_description="Starting in repository root",
                    reasoning="Need to understand project structure before locating the bug",
                    action="find {repo} -type f -name '*.py' | head -30",
                    action_type="exploration",
                ),
                WorkflowStep(
                    env_description="Directory listing available",
                    reasoning="Identify test directory structure",
                    action="ls -la {repo}/tests/",
                    action_type="exploration",
                ),
            ],
            frequency=100,
        ),
        Workflow(
            id="wf-locate-bug",
            description="Locate bug by searching for error keywords from issue",
            category="investigation",
            steps=[
                WorkflowStep(
                    env_description="Issue mentions specific function or error",
                    reasoning="Search for the mentioned function/error in codebase",
                    action="grep -r '{pattern}' {repo} --include='*.py'",
                    action_type="exploration",
                ),
                WorkflowStep(
                    env_description="Search results identify relevant files",
                    reasoning="Examine the most relevant file",
                    action="view({file_path})",
                    action_type="file_view",
                ),
            ],
            frequency=100,
        ),
        Workflow(
            id="wf-fix-verify",
            description="Apply fix and verify with tests",
            category="fix_and_verify",
            steps=[
                WorkflowStep(
                    env_description="Bug location and fix understood",
                    reasoning="Apply minimal fix to resolve the issue",
                    action="str_replace({file_path}, {old_code}, {new_code})",
                    action_type="file_edit",
                ),
                WorkflowStep(
                    env_description="Code modified",
                    reasoning="Run tests to verify fix doesn't break anything",
                    action="cd {repo} && python -m pytest {test_file} -v",
                    action_type="testing",
                ),
            ],
            frequency=100,
        ),
        Workflow(
            id="wf-understand-test",
            description="Understand expected behavior from tests",
            category="investigation",
            steps=[
                WorkflowStep(
                    env_description="Need to understand expected behavior",
                    reasoning="Look at existing tests to understand correct behavior",
                    action="find {repo}/tests -name '*test*.py' -exec grep -l '{function}' {} \\;",
                    action_type="exploration",
                ),
                WorkflowStep(
                    env_description="Test file identified",
                    reasoning="Read test to understand expected inputs/outputs",
                    action="view({test_file})",
                    action_type="file_view",
                ),
            ],
            frequency=50,
        ),
    ]

    def __init__(self, storage_path: Optional[Path] = None):
        self.extractor = WorkflowExtractor()
        self.workflows: List[Workflow] = []
        self.storage_path = storage_path or Path("./swebench_workflows.json")

        # Initialize with base workflows
        self.workflows.extend(self.BASE_WORKFLOWS)

    def ingest_trajectories(
        self, output_jsonl: Path, only_successful: bool = True, use_llm: bool = False
    ):
        """
        Ingest trajectories from OpenHands output and extract workflows.

        Args:
            output_jsonl: Path to output.jsonl file
            only_successful: Only extract from successful trajectories
            use_llm: Use LLM-based extraction (more accurate, requires API)
        """
        parser = OpenHandsEventParser()
        trajectories = parser.parse_output_jsonl(output_jsonl)

        print(f"Loaded {len(trajectories)} trajectories")

        if use_llm:
            new_workflows = self.extractor.extract_llm_based(trajectories)
        else:
            new_workflows = self.extractor.extract_rule_based(
                trajectories, only_successful=only_successful
            )

        # Merge with existing workflows
        self._merge_workflows(new_workflows)

        print(f"Extracted {len(new_workflows)} new workflows")
        print(f"Total workflows: {len(self.workflows)}")

    def _merge_workflows(self, new_workflows: List[Workflow]):
        """Merge new workflows with existing ones."""
        existing_ids = {w.id for w in self.workflows}

        for wf in new_workflows:
            if wf.id in existing_ids:
                # Update existing workflow
                for i, existing in enumerate(self.workflows):
                    if existing.id == wf.id:
                        # Increase frequency
                        existing.frequency += wf.frequency
                        # Add source instances
                        existing.source_instances.extend(wf.source_instances)
                        existing.source_instances = list(set(existing.source_instances))
                        break
            else:
                self.workflows.append(wf)

    def get_relevant_workflows(
        self, task_description: str, max_workflows: int = 5, min_score: float = 0.0
    ) -> List[Workflow]:
        """
        Retrieve workflows relevant to a given task.

        Uses keyword matching and category-based scoring.
        """
        task_lower = task_description.lower()

        scored = []
        for wf in self.workflows:
            score = self._score_relevance(wf, task_lower)
            if score >= min_score:
                scored.append((score, wf))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [wf for _, wf in scored[:max_workflows]]

    def _score_relevance(self, workflow: Workflow, task_lower: str) -> float:
        """Score workflow relevance to a task."""
        score = 0.0
        desc_lower = workflow.description.lower()

        # Keyword matching
        keywords = {
            ("test", "verify", "check"): 2.0,
            ("fix", "bug", "error", "issue"): 2.0,
            ("search", "find", "locate"): 1.5,
            ("understand", "examine", "review"): 1.0,
        }

        for kw_group, weight in keywords.items():
            for kw in kw_group:
                if kw in task_lower:
                    if kw in desc_lower:
                        score += weight

        # Category matching
        if "test" in task_lower and workflow.category in ["testing", "fix_and_verify"]:
            score += 1.5
        if any(w in task_lower for w in ["fix", "bug", "error"]):
            if workflow.category in ["modification", "fix_and_verify"]:
                score += 1.5

        # Frequency bonus (prefer proven workflows)
        score += min(workflow.frequency * 0.1, 2.0)

        return score

    def format_for_prompt(self, workflows: List[Workflow]) -> str:
        """
        Format workflows for inclusion in agent system prompt.

        Follows AWM's format: description + abstracted steps.
        """
        if not workflows:
            return ""

        lines = [
            "## Available Workflows from Past Experience",
            "",
            "The following workflows have been learned from successful bug fixes.",
            "Use them as guidance but adapt to the specific situation.",
            "",
        ]

        for i, wf in enumerate(workflows, 1):
            lines.append(f"### Workflow {i}: {wf.description}")
            lines.append(f"Category: {wf.category}")
            lines.append(f"Used in {wf.frequency} previous tasks")
            lines.append("")

            for j, step in enumerate(wf.steps, 1):
                lines.append(f"**Step {j}:**")
                if step.reasoning:
                    lines.append(f"- Reasoning: {step.reasoning[:200]}")
                lines.append(f"- Action: `{step.action}`")
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def format_as_skill(self, workflows: List[Workflow]) -> str:
        """
        Format workflows as an OpenHands Skill.

        Skills are markdown files that can be loaded into the agent context.
        """
        lines = [
            "# SWE-bench Bug-Fixing Workflows",
            "",
            "This skill provides learned workflows for solving GitHub issues.",
            "",
            "## Trigger",
            "Use these workflows when:",
            "- Fixing bugs in Python repositories",
            "- Solving SWE-bench style issues",
            "- Debugging test failures",
            "",
            "## Workflows",
            "",
        ]

        for wf in workflows:
            lines.append(f"### {wf.description}")
            lines.append(f"**Category:** {wf.category}")
            lines.append("")
            lines.append("```")
            for j, step in enumerate(wf.steps, 1):
                lines.append(f"Step {j}: [{step.action_type}]")
                lines.append(f"  Action: {step.action}")
                if step.reasoning:
                    lines.append(f"  Why: {step.reasoning[:100]}")
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    def save(self):
        """Save workflows to JSON file."""
        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "workflow_count": len(self.workflows),
            "workflows": [w.to_dict() for w in self.workflows],
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(self.workflows)} workflows to {self.storage_path}")

    def load(self):
        """Load workflows from JSON file."""
        if not self.storage_path.exists():
            print(f"No existing workflow file at {self.storage_path}")
            return

        with open(self.storage_path) as f:
            data = json.load(f)

        self.workflows = []
        for wf_data in data.get("workflows", []):
            self.workflows.append(Workflow.from_dict(wf_data))

        print(f"Loaded {len(self.workflows)} workflows from {self.storage_path}")


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="AWM Workflow Extraction for SWE-bench"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract", help="Extract workflows from trajectories"
    )
    extract_parser.add_argument(
        "--input", "-i", required=True, help="Path to output.jsonl"
    )
    extract_parser.add_argument(
        "--output", "-o", default="workflows.json", help="Output path"
    )
    extract_parser.add_argument("--only-successful", action="store_true", default=True)
    extract_parser.add_argument("--use-llm", action="store_true", default=False)

    # Apply command
    apply_parser = subparsers.add_parser("apply", help="Get workflows for a task")
    apply_parser.add_argument(
        "--workflows", "-w", required=True, help="Path to workflows.json"
    )
    apply_parser.add_argument("--task", "-t", required=True, help="Task description")
    apply_parser.add_argument("--format", choices=["prompt", "skill"], default="prompt")
    apply_parser.add_argument(
        "--max", type=int, default=5, help="Max workflows to return"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze workflow statistics"
    )
    analyze_parser.add_argument(
        "--workflows", "-w", required=True, help="Path to workflows.json"
    )

    args = parser.parse_args()

    if args.command == "extract":
        memory = SWEBenchWorkflowMemory(storage_path=Path(args.output))
        memory.ingest_trajectories(
            Path(args.input), only_successful=args.only_successful, use_llm=args.use_llm
        )
        memory.save()

    elif args.command == "apply":
        memory = SWEBenchWorkflowMemory(storage_path=Path(args.workflows))
        memory.load()

        workflows = memory.get_relevant_workflows(args.task, max_workflows=args.max)

        if args.format == "prompt":
            print(memory.format_for_prompt(workflows))
        else:
            print(memory.format_as_skill(workflows))

    elif args.command == "analyze":
        memory = SWEBenchWorkflowMemory(storage_path=Path(args.workflows))
        memory.load()

        print(f"\nTotal workflows: {len(memory.workflows)}")

        # Category breakdown
        categories = defaultdict(int)
        for wf in memory.workflows:
            categories[wf.category] += 1

        print("\nBy category:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

        # Top workflows by frequency
        print("\nTop 10 by frequency:")
        sorted_wfs = sorted(memory.workflows, key=lambda w: w.frequency, reverse=True)
        for i, wf in enumerate(sorted_wfs[:10], 1):
            print(f"  {i}. {wf.description} (freq={wf.frequency})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
