"""
Rule-Based Workflow Extraction (AWM_rule)

This module implements rule-based workflow extraction from agent trajectories.
It uses N-gram pattern mining on action types to identify common sub-routines.

Usage:
    # As module
    from rule_base import RuleBasedExtractor
    extractor = RuleBasedExtractor()
    workflows = extractor.extract(trajectories)

    # As CLI (uses default paths)
    python rule_base.py

    # As CLI (custom paths)
    python rule_base.py --input /path/to/trajectories.jsonl --output /path/to/output.json
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Default Paths
# =============================================================================

DEFAULT_INPUT_PATH = (
    "/home/tsljgj/private/benchmarks/CAWM/trajectories/resolved_trajectories.jsonl"
)
DEFAULT_OUTPUT_DIR = "/home/tsljgj/private/benchmarks/CAWM/workflow"
DEFAULT_OUTPUT_FILE = "rule_workflows.json"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class WorkflowStep:
    """A single step in a workflow."""

    env_description: str  # Description of environment state
    reasoning: str  # Why this action is taken
    action: str  # The abstracted action command
    action_type: str  # Category: exploration, file_view, file_edit, testing, etc.

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "WorkflowStep":
        return cls(**data)


@dataclass
class Workflow:
    """A reusable workflow extracted from trajectories."""

    id: str
    description: str
    category: str
    steps: List[WorkflowStep]
    source_instances: List[str] = field(default_factory=list)
    frequency: int = 1
    pattern: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "description": self.description,
            "category": self.category,
            "steps": [s.to_dict() for s in self.steps],
            "source_instances": self.source_instances,
            "frequency": self.frequency,
            "pattern": list(self.pattern),
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
            pattern=tuple(data.get("pattern", [])),
        )


# =============================================================================
# Action Classification
# =============================================================================

# Patterns for classifying terminal commands
ACTION_PATTERNS = {
    "exploration": [
        r"^find\s",
        r"^grep\s",
        r"^rg\s",
        r"^ag\s",
        r"^ls\s",
        r"^tree\s",
        r"^locate\s",
        r"^which\s",
        r"^whereis\s",
    ],
    "testing": [
        r"^pytest",
        r"^python.*test",
        r"^python.*-m\s+pytest",
        r"^tox\s",
        r"^nosetests",
        r"^unittest",
        r"^make\s+test",
        r"^npm\s+test",
    ],
    "setup": [
        r"^pip\s+install",
        r"^conda\s+install",
        r"^npm\s+install",
        r"^apt\s+install",
        r"^apt-get\s+install",
    ],
    "version_control": [
        r"^git\s",
    ],
    "navigation": [
        r"^cd\s",
        r"^pwd$",
        r"^pushd\s",
        r"^popd$",
    ],
    "execution": [
        r"^python\s+(?!.*test)",
        r"^python3\s+(?!.*test)",
        r"^node\s",
        r"^./",
    ],
}


def classify_terminal_action(command: str) -> str:
    """Classify a terminal command into an action type."""
    command = command.strip()

    for action_type, patterns in ACTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return action_type

    return "terminal"  # Default


def classify_action(event: Dict) -> Optional[str]:
    """Classify an action event into an action type."""
    action = event.get("action")
    if action is None:
        return None
    action_kind = action.get("kind", "")

    if action_kind == "TerminalAction":
        command = action.get("command", "")
        if not command or command.strip() in ["", "C-c", "C-d", "C-z"]:
            return None
        return classify_terminal_action(command)

    elif action_kind == "FileEditorAction":
        cmd = action.get("command", "")
        if cmd == "view":
            return "file_view"
        elif cmd in ["str_replace", "create", "insert"]:
            return "file_edit"
        else:
            return "file_operation"

    elif action_kind == "ThinkAction":
        return None  # Skip think actions

    return "other"


# =============================================================================
# Action Abstraction
# =============================================================================


def abstract_path(path: str) -> str:
    """Abstract a file path to a generic form."""
    # Replace workspace paths
    path = re.sub(r"/workspace/[^/]+/", "{repo}/", path)
    path = re.sub(r"/testbed/[^/]+/", "{repo}/", path)

    # Replace test file patterns
    path = re.sub(r"test_[a-zA-Z_]+\.py", "{test_file}.py", path)
    path = re.sub(r"tests/[a-zA-Z_/]+\.py", "tests/{test_path}.py", path)

    return path


def abstract_command(command: str) -> str:
    """Abstract a command to a generic form."""
    # Abstract paths
    command = re.sub(r"/workspace/[^\s]+", "{repo_path}", command)
    command = re.sub(r"/testbed/[^\s]+", "{repo_path}", command)

    # Abstract grep/find patterns
    command = re.sub(
        r'grep\s+-[a-zA-Z]*\s+"[^"]+"\s+', 'grep {options} "{pattern}" ', command
    )
    command = re.sub(
        r"grep\s+-[a-zA-Z]*\s+'[^']+'\s+", "grep {options} '{pattern}' ", command
    )
    command = re.sub(r'grep\s+"[^"]+"\s+', 'grep "{pattern}" ', command)

    # Abstract pytest patterns
    command = re.sub(r"pytest\s+[^\s]+\.py", "pytest {test_file}", command)
    command = re.sub(
        r"python\s+-m\s+pytest\s+[^\s]+", "python -m pytest {test_path}", command
    )

    # Abstract function/class names in grep
    command = re.sub(r'"def\s+[a-zA-Z_]+"', '"def {function_name}"', command)
    command = re.sub(r'"class\s+[a-zA-Z_]+"', '"class {class_name}"', command)

    return command


def abstract_action(event: Dict) -> str:
    """Create an abstracted version of an action."""
    action = event.get("action")
    if action is None:
        return ""
    action_kind = action.get("kind", "")

    if action_kind == "TerminalAction":
        command = action.get("command", "")
        return abstract_command(command)

    elif action_kind == "FileEditorAction":
        cmd = action.get("command", "")
        path = action.get("path", "")
        abstract_path_str = abstract_path(path)

        if cmd == "view":
            return f"view({abstract_path_str})"
        elif cmd == "str_replace":
            return f"str_replace({abstract_path_str}, {{old_code}}, {{new_code}})"
        elif cmd == "create":
            return f"create({abstract_path_str})"
        else:
            return f"{cmd}({abstract_path_str})"

    return str(action)


# =============================================================================
# Trajectory Parsing
# =============================================================================


@dataclass
class ParsedAction:
    """A parsed action with its classification and abstraction."""

    action_type: str
    abstracted_action: str
    reasoning: str
    env_description: str
    original_event: Dict


def parse_trajectory(trajectory: Dict) -> List[ParsedAction]:
    """Parse a trajectory into a list of classified, abstracted actions."""
    history = trajectory.get("history", [])
    parsed_actions = []

    # Build observation map for env descriptions
    observations = {}
    for event in history:
        if event.get("kind") == "ObservationEvent":
            tool_call_id = event.get("tool_call_id", "")
            observation = event.get("observation")
            if observation is None:
                continue
            content = observation.get("content", [])
            if content and isinstance(content[0], dict):
                observations[tool_call_id] = content[0].get("text", "")[:200]
            elif content:
                observations[tool_call_id] = str(content[0])[:200]

    for event in history:
        if event.get("kind") != "ActionEvent":
            continue

        action_type = classify_action(event)
        if action_type is None or action_type == "navigation":
            continue  # Skip navigation and null actions

        # Extract reasoning from thought
        reasoning = ""
        thoughts = event.get("thought", [])
        if thoughts:
            for t in thoughts:
                if isinstance(t, dict) and t.get("type") == "text":
                    reasoning = t.get("text", "")[:300]
                    break
                elif isinstance(t, str):
                    reasoning = t[:300]
                    break

        # Get environment description from corresponding observation
        tool_call_id = event.get("tool_call_id", "")
        env_description = observations.get(tool_call_id, "")

        parsed_actions.append(
            ParsedAction(
                action_type=action_type,
                abstracted_action=abstract_action(event),
                reasoning=reasoning,
                env_description=env_description,
                original_event=event,
            )
        )

    return parsed_actions


# =============================================================================
# N-gram Pattern Mining
# =============================================================================


def extract_ngrams(
    action_types: List[str], min_n: int = 2, max_n: int = 6
) -> List[Tuple[str, ...]]:
    """Extract n-grams from a sequence of action types."""
    ngrams = []
    for n in range(min_n, min(max_n + 1, len(action_types) + 1)):
        for i in range(len(action_types) - n + 1):
            ngram = tuple(action_types[i : i + n])
            ngrams.append(ngram)
    return ngrams


def find_common_patterns(
    all_action_types: List[List[str]],
    min_frequency: int = 2,
    min_n: int = 2,
    max_n: int = 6,
) -> List[Tuple[Tuple[str, ...], int]]:
    """Find common n-gram patterns across multiple trajectories."""
    pattern_counts = Counter()

    for action_types in all_action_types:
        # Get unique n-grams per trajectory to avoid over-counting
        trajectory_ngrams = set(extract_ngrams(action_types, min_n, max_n))
        pattern_counts.update(trajectory_ngrams)

    # Filter by minimum frequency
    common = [
        (pattern, count)
        for pattern, count in pattern_counts.items()
        if count >= min_frequency
    ]

    # Sort by (frequency * length) to prefer longer common patterns
    common.sort(key=lambda x: x[1] * len(x[0]), reverse=True)

    return common


def deduplicate_patterns(
    patterns: List[Tuple[Tuple[str, ...], int]],
) -> List[Tuple[Tuple[str, ...], int]]:
    """Remove patterns that are subsets of longer patterns."""
    if not patterns:
        return []

    # Sort by length descending
    patterns_sorted = sorted(patterns, key=lambda x: len(x[0]), reverse=True)

    kept = []
    seen_supersets = set()

    for pattern, count in patterns_sorted:
        pattern_str = "-".join(pattern)

        # Check if this pattern is a substring of any kept pattern
        is_subset = False
        for superset in seen_supersets:
            if pattern_str in superset:
                is_subset = True
                break

        if not is_subset:
            kept.append((pattern, count))
            seen_supersets.add(pattern_str)

    return kept


# =============================================================================
# Workflow Generation
# =============================================================================

# Description templates for common patterns
PATTERN_DESCRIPTIONS = {
    ("exploration",): "Search the codebase",
    ("exploration", "file_view"): "Search for and examine relevant files",
    ("exploration", "exploration"): "Comprehensive codebase search",
    ("file_view",): "Examine source code",
    ("file_view", "file_view"): "Review multiple source files",
    ("file_view", "file_edit"): "Examine and modify code",
    ("file_view", "file_edit", "testing"): "Understand, fix, and verify",
    ("file_edit", "testing"): "Apply fix and verify with tests",
    ("file_edit", "testing", "testing"): "Apply fix and run comprehensive tests",
    ("exploration", "file_view", "file_edit"): "Find, understand, and fix code",
    ("exploration", "file_view", "file_edit", "testing"): "Complete bug-fix workflow",
    ("testing",): "Run tests",
    ("testing", "testing"): "Run multiple test suites",
    ("testing", "file_edit"): "Test-driven development iteration",
    ("testing", "file_edit", "testing"): "Debug cycle: test, fix, verify",
}


def generate_pattern_description(pattern: Tuple[str, ...]) -> str:
    """Generate a description for a pattern."""
    # Check exact match
    if pattern in PATTERN_DESCRIPTIONS:
        return PATTERN_DESCRIPTIONS[pattern]

    # Check prefix match
    for known_pattern, desc in PATTERN_DESCRIPTIONS.items():
        if pattern[: len(known_pattern)] == known_pattern:
            return f"{desc} (extended)"

    # Generate from pattern
    parts = []
    if "exploration" in pattern:
        parts.append("search")
    if "file_view" in pattern:
        parts.append("examine")
    if "file_edit" in pattern:
        parts.append("modify")
    if "testing" in pattern:
        parts.append("test")

    if parts:
        return f"Workflow: {', '.join(parts)} code"

    return f"Workflow with {len(pattern)} steps"


def determine_category(pattern: Tuple[str, ...]) -> str:
    """Determine workflow category from pattern."""
    if "file_edit" in pattern and "testing" in pattern:
        return "fix_and_verify"
    elif "file_edit" in pattern:
        return "modification"
    elif "testing" in pattern:
        return "testing"
    elif "file_view" in pattern and "exploration" in pattern:
        return "investigation"
    elif "exploration" in pattern:
        return "exploration"
    else:
        return "general"


def generate_workflow_id(pattern: Tuple[str, ...]) -> str:
    """Generate a unique ID for a workflow."""
    pattern_str = "-".join(pattern)
    hash_val = hashlib.md5(pattern_str.encode()).hexdigest()[:8]
    return f"wf-rule-{hash_val}"


# =============================================================================
# Main Extractor Class
# =============================================================================


class RuleBasedExtractor:
    """
    Rule-based workflow extractor using N-gram pattern mining.

    Parameters:
        min_pattern_length: Minimum number of steps in a workflow (default: 2)
        max_pattern_length: Maximum number of steps in a workflow (default: 6)
        min_frequency: Minimum occurrences to consider a pattern (default: 2)
        only_successful: Only extract from successful trajectories (default: True)
    """

    def __init__(
        self,
        min_pattern_length: int = 2,
        max_pattern_length: int = 6,
        min_frequency: int = 2,
        only_successful: bool = True,
    ):
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.min_frequency = min_frequency
        self.only_successful = only_successful

    def extract(self, trajectories: List[Dict]) -> List[Workflow]:
        """
        Extract workflows from a list of trajectories.

        Args:
            trajectories: List of trajectory dictionaries with 'history' and optionally 'test_result'

        Returns:
            List of Workflow objects
        """
        # Filter successful trajectories if requested
        if self.only_successful:
            trajectories = [
                t for t in trajectories if t.get("test_result", {}).get("git_patch")
            ]

        if not trajectories:
            print("No trajectories to process")
            return []

        print(f"Processing {len(trajectories)} trajectories...")

        # Parse all trajectories
        all_parsed = []
        all_action_types = []
        instance_map = {}  # Map pattern occurrences to instances

        for traj in trajectories:
            instance_id = traj.get("instance_id", "unknown")
            parsed = parse_trajectory(traj)

            if not parsed:
                continue

            all_parsed.append((instance_id, parsed))
            action_types = [p.action_type for p in parsed]
            all_action_types.append(action_types)

            # Track which instances have which patterns
            for ngram in set(
                extract_ngrams(
                    action_types, self.min_pattern_length, self.max_pattern_length
                )
            ):
                if ngram not in instance_map:
                    instance_map[ngram] = []
                instance_map[ngram].append(instance_id)

        # Find common patterns
        # Adjust min_frequency if we have few trajectories
        effective_min_freq = min(self.min_frequency, max(1, len(trajectories) // 2))

        common_patterns = find_common_patterns(
            all_action_types,
            min_frequency=effective_min_freq,
            min_n=self.min_pattern_length,
            max_n=self.max_pattern_length,
        )

        # Deduplicate patterns
        unique_patterns = deduplicate_patterns(common_patterns)

        print(f"Found {len(unique_patterns)} unique patterns")

        # Generate workflows
        workflows = []
        for pattern, frequency in unique_patterns[:20]:  # Limit to top 20
            # Find example steps for this pattern
            steps = self._find_example_steps(pattern, all_parsed)

            if not steps:
                continue

            workflow = Workflow(
                id=generate_workflow_id(pattern),
                description=generate_pattern_description(pattern),
                category=determine_category(pattern),
                steps=steps,
                source_instances=instance_map.get(pattern, [])[:5],  # Limit sources
                frequency=frequency,
                pattern=pattern,
            )
            workflows.append(workflow)

        print(f"Generated {len(workflows)} workflows")
        return workflows

    def _find_example_steps(
        self, pattern: Tuple[str, ...], all_parsed: List[Tuple[str, List[ParsedAction]]]
    ) -> List[WorkflowStep]:
        """Find example steps that match a pattern."""
        for instance_id, parsed in all_parsed:
            action_types = [p.action_type for p in parsed]

            # Find where pattern occurs
            for i in range(len(action_types) - len(pattern) + 1):
                if tuple(action_types[i : i + len(pattern)]) == pattern:
                    # Found match, extract steps
                    steps = []
                    for j, action in enumerate(parsed[i : i + len(pattern)]):
                        step = WorkflowStep(
                            env_description=action.env_description,
                            reasoning=action.reasoning,
                            action=action.abstracted_action,
                            action_type=action.action_type,
                        )
                        steps.append(step)
                    return steps

        return []

    def extract_to_dict(self, trajectories: List[Dict]) -> Dict:
        """Extract workflows and return as serializable dictionary."""
        workflows = self.extract(trajectories)

        return {
            "version": "1.0",
            "extraction_method": "rule_based",
            "created_at": datetime.now().isoformat(),
            "config": {
                "min_pattern_length": self.min_pattern_length,
                "max_pattern_length": self.max_pattern_length,
                "min_frequency": self.min_frequency,
                "only_successful": self.only_successful,
            },
            "workflow_count": len(workflows),
            "workflows": [w.to_dict() for w in workflows],
        }


# =============================================================================
# File I/O
# =============================================================================


def load_trajectories(input_path: str) -> List[Dict]:
    """Load trajectories from a JSONL file."""
    trajectories = []

    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trajectories.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")

    return trajectories


def save_workflows(workflows_dict: Dict, output_path: str):
    """Save workflows to a JSON file."""
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(workflows_dict, f, indent=2)


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Rule-based workflow extraction from agent trajectories"
    )

    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT_PATH,
        help=f"Path to input JSONL file (default: {DEFAULT_INPUT_PATH})",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=f"Path to output JSON file (default: {DEFAULT_OUTPUT_DIR}/{DEFAULT_OUTPUT_FILE})",
    )

    parser.add_argument(
        "--min-length", type=int, default=2, help="Minimum workflow length (default: 2)"
    )

    parser.add_argument(
        "--max-length", type=int, default=6, help="Maximum workflow length (default: 6)"
    )

    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum pattern frequency (default: 2)",
    )

    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed trajectories (default: only successful)",
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        args.output = str(Path(DEFAULT_OUTPUT_DIR) / DEFAULT_OUTPUT_FILE)

    # Load trajectories
    print(f"Loading trajectories from {args.input}...")
    trajectories = load_trajectories(args.input)
    print(f"Loaded {len(trajectories)} trajectories")

    # Extract workflows
    extractor = RuleBasedExtractor(
        min_pattern_length=args.min_length,
        max_pattern_length=args.max_length,
        min_frequency=args.min_frequency,
        only_successful=not args.include_failed,
    )

    result = extractor.extract_to_dict(trajectories)

    # Save
    save_workflows(result, args.output)
    print(f"Saved {result['workflow_count']} workflows to {args.output}")

    # Print summary
    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Workflows extracted: {result['workflow_count']}")

    if result["workflows"]:
        print("\nTop workflows:")
        for i, wf in enumerate(result["workflows"][:5], 1):
            print(
                f"  {i}. {wf['description']} (freq: {wf['frequency']}, steps: {len(wf['steps'])})"
            )


if __name__ == "__main__":
    main()
