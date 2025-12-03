import json
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass

from CAWM.models import Trajectory, Workflow
from CAWM.llm_client import LLMClient
from CAWM.compression import CompressionModule, CompressionStrategy
from CAWM.clustering import ClusteringModule, SimilarityMethod
from CAWM.induction import InductionModule, WorkflowLevel

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    # Compression settings
    compression_strategy: CompressionStrategy = CompressionStrategy.KEY_STEP_EXTRACTION
    
    # Clustering settings
    clustering_method: SimilarityMethod = SimilarityMethod.ACTION_SEQUENCE
    clustering_threshold: float = 0.5
    
    # Induction settings
    workflow_level: WorkflowLevel = WorkflowLevel.GENERAL
    
    # LLM settings (if not passed explicitly)
    llm_model: str = "anthropic/claude-3.5-sonnet"

class CAWMPipeline:
    """
    Orchestrates the complete CAWM workflow extraction process.
    Pipeline: Load -> Compress -> Cluster -> Induce -> Save
    """
    
    def __init__(
        self, 
        llm_client: LLMClient, 
        config: Optional[PipelineConfig] = None
    ):
        self.llm_client = llm_client
        self.config = config or PipelineConfig()
        
        # Initialize modules
        self.compressor = CompressionModule(
            strategy=self.config.compression_strategy,
            llm_client=llm_client
        )
        
        self.clusterer = ClusteringModule(
            method=self.config.clustering_method,
            llm_client=llm_client
        )
        self.clusterer.config.threshold = self.config.clustering_threshold
        
        self.inductor = InductionModule(
            llm_client=llm_client,
            compression_module=self.compressor
        )
        self.inductor.config.level = self.config.workflow_level

    def run(self, trajectories: List[Trajectory]) -> List[Workflow]:
        """Run the pipeline on a list of trajectories."""
        if not trajectories:
            logger.warning("No trajectories provided.")
            return []
            
        logger.info(f"Starting pipeline with {len(trajectories)} trajectories")
        
        # 1. Compress (Optional, can be done inside induction, 
        # but doing it here might help clustering if we used compressed state)
        # For now, we let InductionModule handle compression internally for prompt building,
        # and ClusteringModule handle raw trajectories (or we can compress first).
        # Let's keep raw for clustering to retain max info, unless strategy requires otherwise.
        
        # 2. Cluster
        logger.info("Clustering trajectories...")
        clusters = self.clusterer.cluster(trajectories)
        logger.info(f"Formed {len(clusters)} clusters")
        
        # 3. Induce
        logger.info("Inducing workflows...")
        workflows = self.inductor.induce_from_clusters(clusters, level=self.config.workflow_level)
        logger.info(f"Extracted {len(workflows)} workflows")
        
        return workflows

    def run_from_file(self, input_path: str, output_path: Optional[str] = None) -> List[Workflow]:
        """Run pipeline reading from JSONL file."""
        logger.info(f"Loading trajectories from {input_path}")
        trajectories = Trajectory.load_from_jsonl(input_path)
        
        if not trajectories:
            logger.error("No valid trajectories loaded")
            return []
            
        workflows = self.run(trajectories)
        
        if output_path:
            self._save_workflows(workflows, output_path)
            
        return workflows

    def _save_workflows(self, workflows: List[Workflow], output_path: str):
        """Save workflows to JSON file."""
        data = {
            "workflows": [w.to_dict() for w in workflows],
            "count": len(workflows),
            "config": {
                "level": self.config.workflow_level.name,
                "clustering": self.config.clustering_method.name
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved workflows to {output_path}")
