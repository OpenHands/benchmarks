import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from CAWM.compression import CompressionModule, CompressionStrategy
from CAWM.llm_client import LLMClient
from CAWM.models import Trajectory, TrajectoryCluster, Workflow, WorkflowStep


logger = logging.getLogger(__name__)


class WorkflowLevel(Enum):
    GENERAL = 1  # Cross-project general
    SPECIFIC = 2  # Project/Issue-type specific


@dataclass
class InductionConfig:
    level: WorkflowLevel = WorkflowLevel.GENERAL
    max_workflows: int = 5
    min_steps: int = 2
    max_steps: int = 10


class InductionModule:
    """
    Module for inducing workflows from trajectories using LLM.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[InductionConfig] = None,
        compression_module: Optional[CompressionModule] = None,
    ):
        self.llm_client = llm_client
        self.config = config or InductionConfig()
        self.compression_module = compression_module or CompressionModule(
            strategy=CompressionStrategy.KEY_STEP_EXTRACTION
        )

    def induce(
        self, trajectories: List[Trajectory], level: Optional[WorkflowLevel] = None
    ) -> List[Workflow]:
        """Induce workflows from a list of trajectories."""
        target_level = level or self.config.level

        # 1. Pre-process/Compress trajectories
        compressed_trajs = self.compression_module.compress_batch(trajectories)

        # 2. Prepare Prompt
        prompt = self._build_prompt(compressed_trajs, target_level)

        # 3. Call LLM
        response = self.llm_client.complete(
            prompt, system="You are an expert software engineering workflow analyst."
        )

        # 4. Parse Response
        workflows = self._parse_response(response, target_level)

        # 5. Post-process (add source instances)
        source_ids = [t.instance_id for t in trajectories]
        for wf in workflows:
            wf.source_instances = source_ids[:5]  # Limit to first 5 to avoid bloat

        return workflows

    def induce_from_clusters(
        self, clusters: List[TrajectoryCluster], level: Optional[WorkflowLevel] = None
    ) -> List[Workflow]:
        """Induce workflows from clusters."""
        all_workflows = []
        for cluster in clusters:
            logger.info(
                f"Inducing from cluster {cluster.cluster_id} ({len(cluster)} trajectories)"
            )
            wfs = self.induce(cluster.trajectories, level)
            all_workflows.extend(wfs)
        return all_workflows

    def induce_hierarchical(
        self, trajectories: List[Trajectory]
    ) -> Dict[WorkflowLevel, List[Workflow]]:
        """Induce both general and specific workflows."""
        general = self.induce(trajectories, WorkflowLevel.GENERAL)
        specific = self.induce(trajectories, WorkflowLevel.SPECIFIC)
        return {WorkflowLevel.GENERAL: general, WorkflowLevel.SPECIFIC: specific}

    def _build_prompt(
        self, trajectories: List[Trajectory], level: WorkflowLevel
    ) -> str:
        """Build the prompt for LLM to extract experiences."""

        # Process more trajectories (up to 10) to get diverse experiences
        traj_text = []
        for i, traj in enumerate(trajectories[:10]):
            events_str = []
            for e in traj.events[:20]:  # Limit events per trajectory
                summary = e.get_action_summary()
                events_str.append(f"  - [{e.action_type.value}] {summary}")

            # Include more context: instruction and key actions
            traj_text.append(
                f"=== Trajectory {i + 1}: {traj.instance_id} ===\n"
                f"Problem: {traj.instruction[:500]}\n"
                f"Key Actions:\n" + "\n".join(events_str[:15])
            )

        context = "\n\n".join(traj_text)

        level_instruction = ""
        if level == WorkflowLevel.GENERAL:
            level_instruction = (
                "Extract experiences that could apply to similar problems across different projects.\n"
                "Use placeholders like {file}, {function}, {module} for specific names."
            )
        else:
            level_instruction = (
                "Extract detailed, specific experiences that capture exact patterns and strategies.\n"
                "Include specific error types, module names, and debugging approaches."
            )

        return f"""You are analyzing successful bug-fixing trajectories from AI coding agents.

Your task: Extract SPECIFIC EXPERIENCES and LESSONS LEARNED that would help solve similar problems in the future.

{level_instruction}

IMPORTANT:
- Each experience should be a CONCRETE, ACTIONABLE insight
- NOT generic steps like "write tests" or "read the code"
- Focus on WHAT was learned, not the process of learning it
- Extract 3-5 experiences per trajectory group

Good experience examples:
- "When encountering 'FieldError: Cannot resolve keyword', check if the field name conflicts with a reverse relation defined in related models"
- "For HTTP date parsing issues, ensure the year interpretation follows RFC 7231 - years 00-69 map to 2000-2069, years 70-99 map to 1970-1999"
- "When inner class references fail during migration serialization, use __qualname__ instead of __name__ to get the full dotted path"

Bad experience examples (too generic):
- "First reproduce the bug" (obvious)
- "Write tests before fixing" (generic advice)
- "Read the error message carefully" (not actionable)

Output JSON format:
{{
    "experiences": [
        {{
            "trigger": "When/If <specific situation or error>",
            "insight": "The key discovery or root cause",
            "action": "Specific recommended approach or fix",
            "category": "debugging|refactoring|testing|configuration|api_usage"
        }}
    ]
}}

Trajectories to analyze:
{context}
"""

    def _parse_response(self, response: str, level: WorkflowLevel) -> List[Workflow]:
        """Parse LLM JSON response - handles both old workflow and new experience formats."""
        try:
            data = self.llm_client.parse_structured_response(response)
            workflows = []

            # Handle new "experiences" format
            if "experiences" in data:
                for item in data.get("experiences", []):
                    # Convert experience to Workflow format for compatibility
                    trigger = item.get("trigger", "")
                    insight = item.get("insight", "")
                    action = item.get("action", "")
                    category = item.get("category", "debugging")

                    # Create a single-step workflow representing this experience
                    step = WorkflowStep(
                        env_description=trigger,
                        reasoning=insight,
                        action=action,
                        action_type=category,
                    )

                    # Generate ID from trigger
                    wf_id = f"exp-{level.name.lower()}-{hashlib.md5(trigger.encode()).hexdigest()[:8]}"

                    wf = Workflow(
                        id=wf_id,
                        description=f"{trigger} → {insight[:100]}",
                        category=category,
                        steps=[step],
                        level=level.value,
                        pattern=(category,),
                    )
                    workflows.append(wf)

            # Handle old "workflows" format for backward compatibility
            elif "workflows" in data:
                for item in data.get("workflows", []):
                    steps = []
                    for s in item.get("steps", []):
                        steps.append(
                            WorkflowStep(
                                env_description=s.get("env_description", ""),
                                reasoning=s.get("reasoning", ""),
                                action=s.get("action", ""),
                                action_type=s.get("action_type", "other"),
                            )
                        )

                    wf_id = f"wf-{level.name.lower()}-{hashlib.md5(item.get('name', '').encode()).hexdigest()[:8]}"

                    wf = Workflow(
                        id=wf_id,
                        description=item.get("description", item.get("name", "")),
                        category=item.get("category", "general"),
                        steps=steps,
                        level=level.value,
                        pattern=tuple(s.action_type for s in steps),
                    )
                    workflows.append(wf)

            return workflows

        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            logger.debug(f"Raw response: {response[:500]}")
            return []
