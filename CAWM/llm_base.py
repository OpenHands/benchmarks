"""
LLM-Based Workflow Extraction (AWM_lm)

This module implements LLM-based workflow extraction from agent trajectories.
It uses an LLM to identify semantically meaningful sub-routines and abstract them.

Usage:
    # As module
    from llm_base import LLMBasedExtractor
    extractor = LLMBasedExtractor(provider="openai", model="gpt-4")
    workflows = extractor.extract(trajectories)

    # As CLI (uses default paths)
    python llm_base.py

    # As CLI (custom paths)
    python llm_base.py --input /path/to/trajectories.jsonl --output /path/to/output.json

    # With mock (no API needed)
    python llm_base.py --mock
"""

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Try to import LLM libraries
try:
    import openai  # type: ignore[import-untyped]

    HAS_OPENAI = True
except ImportError:
    openai = None  # type: ignore[assignment]
    HAS_OPENAI = False

try:
    import anthropic  # type: ignore[import-untyped]

    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    HAS_ANTHROPIC = False


# =============================================================================
# Default Paths
# =============================================================================

DEFAULT_INPUT_PATH = (
    "/home/tsljgj/private/benchmarks/CAWM/trajectories/resolved_trajectories.jsonl"
)
DEFAULT_OUTPUT_DIR = "/home/tsljgj/private/benchmarks/CAWM/workflow"
DEFAULT_OUTPUT_FILE = "llm_workflows.json"


# =============================================================================
# Data Classes (Compatible with rule_base.py)
# =============================================================================


@dataclass
class WorkflowStep:
    """A single step in a workflow."""

    env_description: str
    reasoning: str
    action: str
    action_type: str

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
# Prompt Templates
# =============================================================================

WORKFLOW_EXTRACTION_PROMPT = """You are an expert at analyzing software engineering workflows. Given a set of bug-fixing trajectories, extract common reusable workflows.

## Task
Analyze the trajectories below and extract 3-8 common workflows that could help solve similar bugs.

## Requirements
1. **Sub-routine level**: Extract workflows at sub-task granularity, not full solutions
   - Good: "Locate function definition in codebase"
   - Bad: "Fix the entire bug"

2. **Abstraction**: Replace specific values with placeholders:
   - Paths: `/workspace/django/` → `{{repo}}/`
   - Functions: `my_function` → `{{function_name}}`
   - Files: `test_models.py` → `{{test_file}}`
   - Patterns: `"error message"` → `{{search_pattern}}`

3. **Minimum 2 steps** per workflow

4. **Categories**: Use one of: exploration, investigation, modification, fix_and_verify, testing

## Output Format
For each workflow, use EXACTLY this format:

```
WORKFLOW: [Name]
CATEGORY: [category]
DESCRIPTION: [When to use this workflow]

STEP 1:
ENV: [Environment state before this step]
REASONING: [Why take this action]
ACTION: [Abstracted command]
ACTION_TYPE: [exploration|file_view|file_edit|testing|terminal]

STEP 2:
ENV: [Environment state]
REASONING: [Why]
ACTION: [Command]
ACTION_TYPE: [type]

---
```

## Trajectories to Analyze

{trajectories}

## Extract Workflows
Now extract the common workflows. Remember to abstract specific values and focus on reusable patterns."""


SINGLE_TRAJECTORY_PROMPT = """Analyze this bug-fixing trajectory and extract 1-3 reusable workflow patterns.

## Task: {task_description}
## Repository: {repository}

## Actions Taken:
{actions}

## Requirements
1. Extract workflows at sub-task level (not the full solution)
2. Abstract specific values: paths → `{{repo}}`, functions → `{{function}}`, etc.
3. Each workflow should have 2-5 steps
4. Focus on patterns useful for similar bugs

## Output Format
```
WORKFLOW: [Name]
CATEGORY: [exploration|investigation|modification|fix_and_verify|testing]
DESCRIPTION: [When to use]

STEP 1:
ENV: [State]
REASONING: [Why]
ACTION: [Abstracted command]
ACTION_TYPE: [type]

STEP 2:
...

---
```

Extract workflows:"""


# =============================================================================
# LLM Client
# =============================================================================


class LLMClient:
    """Wrapper for LLM API calls."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key
        if api_key:
            self.api_key = api_key
        elif provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            self.api_key = None

        self._validate_setup()

    def _validate_setup(self):
        """Validate LLM setup."""
        if self.provider == "openai" and not HAS_OPENAI:
            raise ImportError("openai package not installed. Run: pip install openai")
        if self.provider == "anthropic" and not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        if not self.api_key:
            raise ValueError(
                f"API key not found. Set {self.provider.upper()}_API_KEY environment variable"
            )

    def complete(self, prompt: str) -> str:
        """Get completion from LLM."""
        if self.provider == "openai":
            return self._complete_openai(prompt)
        elif self.provider == "anthropic":
            return self._complete_anthropic(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _complete_openai(self, prompt: str) -> str:
        """OpenAI API completion."""
        client = openai.OpenAI(api_key=self.api_key)  # type: ignore[union-attr]

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing software engineering workflows.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content
        return content if content is not None else ""

    def _complete_anthropic(self, prompt: str) -> str:
        """Anthropic API completion."""
        client = anthropic.Anthropic(api_key=self.api_key)  # type: ignore[union-attr]

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system="You are an expert at analyzing software engineering workflows.",
            messages=[{"role": "user", "content": prompt}],
        )

        # Get text from first content block (should be TextBlock)
        first_block = response.content[0]
        if hasattr(first_block, "text"):
            return first_block.text  # type: ignore[union-attr]
        return ""


# =============================================================================
# Trajectory Formatter
# =============================================================================


class TrajectoryFormatter:
    """Format trajectories for LLM consumption."""

    def __init__(self, max_steps: int = 30, max_trajectories: int = 5):
        self.max_steps = max_steps
        self.max_trajectories = max_trajectories

    def format_single(self, trajectory: Dict) -> str:
        """Format a single trajectory."""
        instance_id = trajectory.get("instance_id", "unknown")
        instruction = trajectory.get("instruction", "")
        repo = instance_id.split("__")[0] if "__" in instance_id else "unknown"

        lines = [
            f"### {instance_id}",
            f"Repository: {repo}",
            f"Task: {instruction[:400]}",
            "",
            "Actions:",
        ]

        history = trajectory.get("history", [])
        step = 0

        for event in history:
            if step >= self.max_steps:
                lines.append(f"... ({len(history) - step} more steps)")
                break

            if event.get("kind") != "ActionEvent":
                continue

            formatted = self._format_action(event)
            if formatted:
                step += 1
                lines.append(f"{step}. {formatted}")

        return "\n".join(lines)

    def _format_action(self, event: Dict) -> Optional[str]:
        """Format a single action."""
        action = event.get("action")
        if action is None:
            return None
        action_kind = action.get("kind", "")

        # Get thought
        thought = ""
        thoughts = event.get("thought", [])
        if thoughts:
            for t in thoughts:
                if isinstance(t, dict):
                    thought = t.get("text", "")[:150]
                    break

        if action_kind == "TerminalAction":
            cmd = action.get("command", "")
            if not cmd or cmd.strip() in ["", "C-c", "C-d"]:
                return None
            result = f"[Terminal] {cmd}"
            if thought:
                result = f"[Terminal] # {thought}\n   {cmd}"
            return result

        elif action_kind == "FileEditorAction":
            cmd = action.get("command", "")
            path = action.get("path", "")

            if cmd == "view":
                return f"[View] {path}"
            elif cmd == "str_replace":
                old = action.get("old_str", "")[:80]
                new = action.get("new_str", "")[:80]
                return f"[Edit] {path}\n   -{old}...\n   +{new}..."
            else:
                return f"[{cmd}] {path}"

        return None

    def format_batch(self, trajectories: List[Dict]) -> str:
        """Format multiple trajectories."""
        # Filter successful
        successful = [
            t for t in trajectories if t.get("test_result", {}).get("git_patch")
        ]
        selected = (
            successful[: self.max_trajectories]
            if successful
            else trajectories[: self.max_trajectories]
        )

        formatted = []
        for i, traj in enumerate(selected, 1):
            formatted.append(f"## Trajectory {i}")
            formatted.append(self.format_single(traj))
            formatted.append("\n" + "=" * 40 + "\n")

        return "\n".join(formatted)


# =============================================================================
# Response Parser
# =============================================================================


class WorkflowParser:
    """Parse LLM responses into Workflow objects."""

    def parse(self, response: str) -> List[Workflow]:
        """Parse LLM response into workflows."""
        workflows = []
        blocks = self._split_blocks(response)

        for block in blocks:
            try:
                workflow = self._parse_block(block)
                if workflow and len(workflow.steps) >= 2:
                    workflows.append(workflow)
            except Exception as e:
                print(f"Warning: Failed to parse block: {e}")

        return workflows

    def _split_blocks(self, response: str) -> List[str]:
        """Split response into workflow blocks."""
        # Try various delimiters
        if "---" in response:
            blocks = response.split("---")
        elif "WORKFLOW:" in response:
            parts = response.split("WORKFLOW:")
            blocks = ["WORKFLOW:" + p for p in parts[1:]]
        else:
            blocks = [response]

        return [b.strip() for b in blocks if b.strip() and "WORKFLOW" in b.upper()]

    def _parse_block(self, block: str) -> Optional[Workflow]:
        """Parse a single workflow block."""
        lines = block.split("\n")

        # Extract metadata
        name = self._extract_field(lines, "WORKFLOW:")
        if not name:
            return None

        category = self._extract_field(lines, "CATEGORY:") or "general"
        description = self._extract_field(lines, "DESCRIPTION:") or name

        # Extract steps
        steps = self._extract_steps(block)
        if not steps:
            return None

        # Generate ID
        workflow_id = self._generate_id(name, steps)
        pattern = tuple(s.action_type for s in steps)

        return Workflow(
            id=workflow_id,
            description=description,
            category=category.lower().strip(),
            steps=steps,
            frequency=1,
            pattern=pattern,
        )

    def _extract_field(self, lines: List[str], field: str) -> Optional[str]:
        """Extract a field value."""
        for line in lines:
            if line.strip().upper().startswith(field.upper()):
                return line.split(":", 1)[-1].strip()
        return None

    def _extract_steps(self, block: str) -> List[WorkflowStep]:
        """Extract steps from block."""
        steps = []
        step_matches = list(re.finditer(r"STEP\s*(\d+):", block, re.IGNORECASE))

        for i, match in enumerate(step_matches):
            start = match.end()
            end = (
                step_matches[i + 1].start() if i + 1 < len(step_matches) else len(block)
            )

            content = block[start:end]
            step = self._parse_step(content)
            if step:
                steps.append(step)

        return steps

    def _parse_step(self, content: str) -> Optional[WorkflowStep]:
        """Parse step content."""
        env = ""
        reasoning = ""
        action = ""
        action_type = "terminal"

        for line in content.split("\n"):
            line = line.strip()
            upper = line.upper()

            if upper.startswith("ENV:"):
                env = line.split(":", 1)[-1].strip()
            elif upper.startswith("REASONING:"):
                reasoning = line.split(":", 1)[-1].strip()
            elif upper.startswith("ACTION:"):
                action = line.split(":", 1)[-1].strip()
            elif upper.startswith("ACTION_TYPE:"):
                action_type = line.split(":", 1)[-1].strip().lower()

        if not action:
            return None

        return WorkflowStep(
            env_description=env,
            reasoning=reasoning,
            action=action,
            action_type=action_type,
        )

    def _generate_id(self, name: str, steps: List[WorkflowStep]) -> str:
        """Generate workflow ID."""
        types = "-".join(s.action_type for s in steps)
        content = f"{name}-{types}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"wf-llm-{hash_val}"


# =============================================================================
# Main Extractor Class
# =============================================================================


class LLMBasedExtractor:
    """
    LLM-based workflow extractor.

    Parameters:
        provider: LLM provider ("openai" or "anthropic")
        model: Model name (e.g., "gpt-4", "claude-3-opus-20240229")
        api_key: API key (or set via environment variable)
        temperature: Sampling temperature (default: 0.0)
        max_trajectories: Max trajectories per batch (default: 5)
        only_successful: Only extract from successful trajectories (default: True)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_trajectories: int = 5,
        only_successful: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_trajectories = max_trajectories
        self.only_successful = only_successful

        self.llm: Optional[LLMClient] = None  # Lazy initialization
        self.formatter = TrajectoryFormatter(max_trajectories=max_trajectories)
        self.parser = WorkflowParser()

    def _init_llm(self):
        """Initialize LLM client (lazy)."""
        if self.llm is None:
            self.llm = LLMClient(
                provider=self.provider,
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
            )

    def extract(self, trajectories: List[Dict]) -> List[Workflow]:
        """
        Extract workflows from trajectories.

        Args:
            trajectories: List of trajectory dictionaries

        Returns:
            List of Workflow objects
        """
        # Filter successful if requested
        if self.only_successful:
            trajectories = [
                t for t in trajectories if t.get("test_result", {}).get("git_patch")
            ]

        if not trajectories:
            print("No trajectories to process")
            return []

        print(f"Processing {len(trajectories)} trajectories...")

        # Initialize LLM
        self._init_llm()

        # Group by repository for better extraction
        repo_groups = {}
        for traj in trajectories:
            instance_id = traj.get("instance_id", "unknown")
            repo = instance_id.split("__")[0] if "__" in instance_id else "unknown"
            if repo not in repo_groups:
                repo_groups[repo] = []
            repo_groups[repo].append(traj)

        # Extract from each group
        all_workflows = []
        for repo, group in repo_groups.items():
            print(f"  Extracting from {len(group)} {repo} trajectories...")

            # Format trajectories
            formatted = self.formatter.format_batch(group)
            prompt = WORKFLOW_EXTRACTION_PROMPT.format(trajectories=formatted)

            # Get LLM response
            try:
                assert self.llm is not None, "LLM client not initialized"
                response = self.llm.complete(prompt)
                workflows = self.parser.parse(response)

                # Add metadata
                for wf in workflows:
                    wf.source_instances = [t.get("instance_id", "") for t in group[:3]]

                all_workflows.extend(workflows)
                print(f"    Extracted {len(workflows)} workflows")

            except Exception as e:
                print(f"    Error extracting from {repo}: {e}")

        # Deduplicate
        workflows = self._deduplicate(all_workflows)
        print(f"Total unique workflows: {len(workflows)}")

        return workflows

    def extract_single(self, trajectory: Dict) -> List[Workflow]:
        """
        Extract workflows from a single trajectory (online mode).

        Args:
            trajectory: Single trajectory dictionary

        Returns:
            List of Workflow objects
        """
        instance_id = trajectory.get("instance_id", "unknown")
        repo = instance_id.split("__")[0] if "__" in instance_id else "unknown"
        task = trajectory.get("instruction", "Fix a bug")

        # Initialize LLM
        self._init_llm()

        # Format actions
        actions = self.formatter.format_single(trajectory)

        # Build prompt
        prompt = SINGLE_TRAJECTORY_PROMPT.format(
            task_description=task[:500], repository=repo, actions=actions
        )

        # Get response
        try:
            assert self.llm is not None, "LLM client not initialized"
            response = self.llm.complete(prompt)
            workflows = self.parser.parse(response)

            for wf in workflows:
                wf.source_instances = [instance_id]

            return workflows

        except Exception as e:
            print(f"Error extracting from {instance_id}: {e}")
            return []

    def _deduplicate(self, workflows: List[Workflow]) -> List[Workflow]:
        """Remove duplicate workflows."""
        seen = set()
        unique = []

        for wf in workflows:
            # Normalize description
            key = wf.description.lower().strip()
            key = re.sub(r"\s+", " ", key)

            if key not in seen:
                seen.add(key)
                unique.append(wf)
            else:
                # Merge frequency
                for existing in unique:
                    if existing.description.lower().strip() == key:
                        existing.frequency += wf.frequency
                        break

        return unique

    def extract_to_dict(self, trajectories: List[Dict]) -> Dict:
        """Extract workflows and return as dictionary."""
        workflows = self.extract(trajectories)

        return {
            "version": "1.0",
            "extraction_method": "llm_based",
            "llm_provider": self.provider,
            "llm_model": self.model,
            "created_at": datetime.now().isoformat(),
            "config": {
                "temperature": self.temperature,
                "max_trajectories": self.max_trajectories,
                "only_successful": self.only_successful,
            },
            "workflow_count": len(workflows),
            "workflows": [w.to_dict() for w in workflows],
        }


# =============================================================================
# Mock Extractor (for testing without API)
# =============================================================================


class MockLLMExtractor(LLMBasedExtractor):
    """Mock extractor that returns pre-defined workflows for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mock_response = self._get_mock_response()

    def _init_llm(self):
        """Skip LLM initialization."""
        pass

    def extract(self, trajectories: List[Dict]) -> List[Workflow]:
        """Return mock workflows."""
        if self.only_successful:
            trajectories = [
                t for t in trajectories if t.get("test_result", {}).get("git_patch")
            ]

        if not trajectories:
            return []

        print(f"[MOCK] Processing {len(trajectories)} trajectories...")

        workflows = self.parser.parse(self._mock_response)

        for wf in workflows:
            wf.source_instances = [t.get("instance_id", "") for t in trajectories[:3]]

        print(f"[MOCK] Generated {len(workflows)} workflows")
        return workflows

    def _get_mock_response(self) -> str:
        return """
WORKFLOW: Search for function definition
CATEGORY: exploration
DESCRIPTION: Locate where a specific function is defined in the codebase

STEP 1:
ENV: Starting at repository root
REASONING: Use grep to find all occurrences of the function name
ACTION: grep -r "{{function_name}}" {{repo}} --include="*.py"
ACTION_TYPE: exploration

STEP 2:
ENV: Found multiple files containing the function
REASONING: Filter to find the actual definition using 'def' keyword
ACTION: grep -r "def {{function_name}}" {{repo}} --include="*.py"
ACTION_TYPE: exploration

---

WORKFLOW: Examine and understand code
CATEGORY: investigation
DESCRIPTION: Read source code to understand the bug and expected behavior

STEP 1:
ENV: Located the file with the bug
REASONING: View the file to understand the code structure
ACTION: view({{source_file}})
ACTION_TYPE: file_view

STEP 2:
ENV: Reading the source code
REASONING: Find related tests to understand expected behavior
ACTION: find {{repo}}/tests -name "*test*.py" -exec grep -l "{{function_name}}" {{}} \\;
ACTION_TYPE: exploration

STEP 3:
ENV: Found test file
REASONING: Read tests to understand correct behavior
ACTION: view({{test_file}})
ACTION_TYPE: file_view

---

WORKFLOW: Apply fix and verify
CATEGORY: fix_and_verify
DESCRIPTION: Make code changes and run tests to verify the fix

STEP 1:
ENV: Understood the bug and what needs to change
REASONING: Apply minimal change to fix the issue
ACTION: str_replace({{source_file}}, {{buggy_code}}, {{fixed_code}})
ACTION_TYPE: file_edit

STEP 2:
ENV: Code has been modified
REASONING: Run specific failing test to verify fix
ACTION: cd {{repo}} && python -m pytest {{test_file}} -v -k "{{test_name}}"
ACTION_TYPE: testing

STEP 3:
ENV: Specific test passed
REASONING: Run broader test suite to check for regressions
ACTION: cd {{repo}} && python -m pytest {{test_dir}} -v
ACTION_TYPE: testing

---

WORKFLOW: Create reproduction script
CATEGORY: investigation
DESCRIPTION: Create a minimal script to reproduce the bug before fixing

STEP 1:
ENV: Read the issue description
REASONING: Create a script to verify the bug exists
ACTION: create({{repo}}/reproduce.py)
ACTION_TYPE: file_edit

STEP 2:
ENV: Script created
REASONING: Run to confirm the bug
ACTION: cd {{repo}} && python reproduce.py
ACTION_TYPE: testing

---
"""


# =============================================================================
# File I/O
# =============================================================================


def load_trajectories(input_path: str) -> List[Dict]:
    """Load trajectories from JSONL file."""
    trajectories = []

    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    trajectories.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")

    return trajectories


def save_workflows(data: Dict, output_path: str):
    """Save workflows to JSON file."""
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="LLM-based workflow extraction from agent trajectories"
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
        "--provider",
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider (default: openai)",
    )

    parser.add_argument("--model", default="gpt-4", help="Model name (default: gpt-4)")

    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=5,
        help="Max trajectories per batch (default: 5)",
    )

    parser.add_argument(
        "--include-failed", action="store_true", help="Include failed trajectories"
    )

    parser.add_argument(
        "--mock", action="store_true", help="Use mock LLM (for testing without API)"
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        args.output = str(Path(DEFAULT_OUTPUT_DIR) / DEFAULT_OUTPUT_FILE)

    # Load trajectories
    print(f"Loading trajectories from {args.input}...")
    trajectories = load_trajectories(args.input)
    print(f"Loaded {len(trajectories)} trajectories")

    # Create extractor
    if args.mock:
        print("[Using mock LLM - no API calls will be made]")
        extractor = MockLLMExtractor(
            max_trajectories=args.max_trajectories,
            only_successful=not args.include_failed,
        )
    else:
        extractor = LLMBasedExtractor(
            provider=args.provider,
            model=args.model,
            max_trajectories=args.max_trajectories,
            only_successful=not args.include_failed,
        )

    # Extract
    result = extractor.extract_to_dict(trajectories)

    # Save
    save_workflows(result, args.output)
    print(f"Saved {result['workflow_count']} workflows to {args.output}")

    # Summary
    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Method: LLM-based ({args.provider}/{args.model})")
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Workflows extracted: {result['workflow_count']}")

    if result["workflows"]:
        print("\nExtracted workflows:")
        for i, wf in enumerate(result["workflows"], 1):
            print(f"  {i}. {wf['description']}")
            print(f"     Category: {wf['category']}, Steps: {len(wf['steps'])}")


if __name__ == "__main__":
    main()
