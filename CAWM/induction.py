from enum import Enum
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging
import hashlib

from CAWM.models import Trajectory, Workflow, TrajectoryCluster, WorkflowStep
from CAWM.llm_client import LLMClient
from CAWM.compression import CompressionModule, CompressionStrategy

logger = logging.getLogger(__name__)

class WorkflowLevel(Enum):
    GENERAL = 1   # Cross-project general
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
        compression_module: Optional[CompressionModule] = None
    ):
        self.llm_client = llm_client
        self.config = config or InductionConfig()
        self.compression_module = compression_module or CompressionModule(strategy=CompressionStrategy.KEY_STEP_EXTRACTION)

    def induce(self, trajectories: List[Trajectory], level: Optional[WorkflowLevel] = None) -> List[Workflow]:
        """Induce workflows from a list of trajectories."""
        target_level = level or self.config.level
        
        # 1. Pre-process/Compress trajectories
        compressed_trajs = self.compression_module.compress_batch(trajectories)
        
        # 2. Prepare Prompt
        prompt = self._build_prompt(compressed_trajs, target_level)
        
        # 3. Call LLM
        response = self.llm_client.complete(prompt, system="You are an expert software engineering workflow analyst.")
        
        # 4. Parse Response
        workflows = self._parse_response(response, target_level)
        
        # 5. Post-process (add source instances)
        source_ids = [t.instance_id for t in trajectories]
        for wf in workflows:
            wf.source_instances = source_ids[:5] # Limit to first 5 to avoid bloat
            
        return workflows

    def induce_from_clusters(self, clusters: List[TrajectoryCluster], level: Optional[WorkflowLevel] = None) -> List[Workflow]:
        """Induce workflows from clusters."""
        all_workflows = []
        for cluster in clusters:
            logger.info(f"Inducing from cluster {cluster.cluster_id} ({len(cluster)} trajectories)")
            wfs = self.induce(cluster.trajectories, level)
            all_workflows.extend(wfs)
        return all_workflows

    def induce_hierarchical(self, trajectories: List[Trajectory]) -> Dict[WorkflowLevel, List[Workflow]]:
        """Induce both general and specific workflows."""
        general = self.induce(trajectories, WorkflowLevel.GENERAL)
        specific = self.induce(trajectories, WorkflowLevel.SPECIFIC)
        return {
            WorkflowLevel.GENERAL: general,
            WorkflowLevel.SPECIFIC: specific
        }

    def _build_prompt(self, trajectories: List[Trajectory], level: WorkflowLevel) -> str:
        """Build the prompt for LLM."""
        
        traj_text = []
        for i, traj in enumerate(trajectories[:5]): # Limit context window
            events_str = []
            for e in traj.events:
                summary = e.get_action_summary()
                events_str.append(f"- [{e.action_type.value}] {summary}")
            
            traj_text.append(f"Trajectory {i+1} ({traj.instance_id}):\nTask: {traj.instruction[:200]}...\nSteps:\n" + "\n".join(events_str))
            
        context = "\n\n".join(traj_text)
        
        level_instruction = ""
        if level == WorkflowLevel.GENERAL:
            level_instruction = (
                "Extract HIGH-LEVEL, GENERAL workflows applicable across any software project.\n"
                "Abstract away all specific file names, function names, and paths (use placeholders like {{file}}, {{func}})"
            )
        else:
            level_instruction = (
                "Extract SPECIFIC workflows that capture detailed coding patterns or debugging strategies.\n"
                "You can retain common patterns but abstract specific instance values."
            )

        return f"""
Analyze the following execution trajectories of AI agents fixing software bugs.
{level_instruction}

Output the workflows in the following JSON format:
{{
    "workflows": [
        {{
            "name": "Workflow Name",
            "category": "exploration|investigation|modification|fix_and_verify|testing",
            "description": "When to use this",
            "steps": [
                {{
                    "env_description": "State before step",
                    "reasoning": "Why take this step",
                    "action": "Abstracted command",
                    "action_type": "exploration|file_view|file_edit|testing|terminal"
                }}
            ]
        }}
    ]
}}

Trajectories:
{context}
"""

    def _parse_response(self, response: str, level: WorkflowLevel) -> List[Workflow]:
        """Parse LLM JSON response."""
        try:
            data = self.llm_client.parse_structured_response(response)
            workflows = []
            
            for item in data.get("workflows", []):
                steps = []
                for s in item.get("steps", []):
                    steps.append(WorkflowStep(
                        env_description=s.get("env_description", ""),
                        reasoning=s.get("reasoning", ""),
                        action=s.get("action", ""),
                        action_type=s.get("action_type", "other")
                    ))
                
                # Generate ID
                wf_id = f"wf-{level.name.lower()}-{hashlib.md5(item.get('name', '').encode()).hexdigest()[:8]}"
                
                wf = Workflow(
                    id=wf_id,
                    description=item.get("description", item.get("name", "")),
                    category=item.get("category", "general"),
                    steps=steps,
                    level=level.value,
                    pattern=tuple(s.action_type for s in steps)
                )
                workflows.append(wf)
                
            return workflows
            
        except Exception as e:
            logger.error(f"Error parsing workflow response: {e}")
            return []
