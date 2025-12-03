import json
import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from enum import Enum

from CAWM.models import Trajectory, Workflow, TrajectoryCluster
from CAWM.llm_client import LLMClient
from CAWM.compression import CompressionModule, CompressionStrategy
from CAWM.clustering import ClusteringModule, SimilarityMethod
from CAWM.induction import InductionModule, WorkflowLevel

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class OutputManager:
    """
    Manages standardized output files for the CAWM pipeline.

    All output files are created at the start of the pipeline with placeholder
    content and updated progressively as each stage completes. This ensures
    consistent output structure regardless of pipeline configuration.

    Output Files:
        - workflows.json: Final extracted experiences/workflows
        - summary.txt: Human-readable summary
        - clusters.json: Clustering details
        - induction_details.json: Per-cluster induction details
        - pipeline_stats.json: Timing and statistics
    """

    # Standard output file names
    FILES = {
        "workflows": "workflows.json",
        "summary": "summary.txt",
        "clusters": "clusters.json",
        "induction": "induction_details.json",
        "stats": "pipeline_stats.json",
    }

    def __init__(self, output_dir: Path, config: "PipelineConfig"):
        self.output_dir = output_dir
        self.config = config
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(self, filename: str, data: Dict[str, Any]):
        """Write JSON data to file."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _write_text(self, filename: str, content: str):
        """Write text content to file."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            f.write(content)

    def initialize_all(self, num_trajectories: int):
        """
        Initialize all output files with placeholder content at pipeline start.
        This ensures all files exist even if pipeline fails partway through.
        """
        timestamp = datetime.now().isoformat()

        # 1. Initialize workflows.json
        self._write_json(self.FILES["workflows"], {
            "metadata": {
                "generated_at": timestamp,
                "status": StageStatus.PENDING.value,
                "num_trajectories": num_trajectories,
                "config": {
                    "compression": self.config.compression_strategy.value,
                    "clustering": self.config.clustering_method.value,
                    "threshold": self.config.clustering_threshold,
                    "level": self.config.workflow_level.name,
                    "model": self.config.llm_model,
                }
            },
            "workflows": [],
            "count": 0
        })

        # 2. Initialize summary.txt
        self._write_text(self.FILES["summary"],
            f"CAWM Workflow Extraction Summary\n"
            f"{'=' * 40}\n\n"
            f"Status: {StageStatus.PENDING.value}\n"
            f"Generated: {timestamp}\n"
            f"Trajectories: {num_trajectories}\n\n"
            f"Pipeline is initializing...\n"
        )

        # 3. Initialize clusters.json
        self._write_json(self.FILES["clusters"], {
            "metadata": {
                "generated_at": timestamp,
                "status": StageStatus.PENDING.value,
                "clustering_method": self.config.clustering_method.value,
                "threshold": self.config.clustering_threshold,
                "num_clusters": 0,
                "total_trajectories": num_trajectories
            },
            "clusters": []
        })

        # 4. Initialize induction_details.json
        self._write_json(self.FILES["induction"], {
            "metadata": {
                "generated_at": timestamp,
                "status": StageStatus.PENDING.value,
                "workflow_level": self.config.workflow_level.name,
                "num_clusters_processed": 0,
                "total_experiences": 0
            },
            "cluster_results": []
        })

        # 5. Initialize pipeline_stats.json
        self._write_json(self.FILES["stats"], {
            "status": StageStatus.PENDING.value,
            "start_time": timestamp,
            "end_time": None,
            "total_duration_seconds": 0,
            "stages": {
                "input": {
                    "status": StageStatus.COMPLETED.value,
                    "num_trajectories": num_trajectories
                },
                "clustering": {
                    "status": StageStatus.PENDING.value,
                    "method": self.config.clustering_method.value,
                    "threshold": self.config.clustering_threshold,
                    "num_clusters": 0,
                    "cluster_sizes": [],
                    "duration_seconds": 0
                },
                "induction": {
                    "status": StageStatus.PENDING.value,
                    "level": self.config.workflow_level.name,
                    "num_experiences": 0,
                    "experiences_per_cluster": [],
                    "duration_seconds": 0
                }
            },
            "config": {
                "compression_strategy": self.config.compression_strategy.value,
                "clustering_method": self.config.clustering_method.value,
                "clustering_threshold": self.config.clustering_threshold,
                "workflow_level": self.config.workflow_level.name,
                "llm_model": self.config.llm_model
            }
        })

        logger.info(f"Initialized all output files in {self.output_dir}")

    def update_clustering(self, clusters: List[TrajectoryCluster], duration: float):
        """Update clusters.json after clustering stage completes."""
        timestamp = datetime.now().isoformat()

        clusters_data = {
            "metadata": {
                "generated_at": timestamp,
                "status": StageStatus.COMPLETED.value,
                "clustering_method": self.config.clustering_method.value,
                "threshold": self.config.clustering_threshold,
                "num_clusters": len(clusters),
                "total_trajectories": sum(len(c) for c in clusters)
            },
            "clusters": []
        }

        for cluster in clusters:
            cluster_info = {
                "cluster_id": cluster.cluster_id,
                "label": cluster.label,
                "size": len(cluster),
                "trajectories": [
                    {
                        "instance_id": t.instance_id,
                        "repository": t.repository,
                        "instruction_preview": t.instruction[:200] + "..." if len(t.instruction) > 200 else t.instruction,
                        "num_events": len(t.events),
                        "action_types": list(set(e.action_type.name for e in t.events))
                    }
                    for t in cluster.trajectories
                ]
            }
            clusters_data["clusters"].append(cluster_info)

        self._write_json(self.FILES["clusters"], clusters_data)

        # Also update pipeline_stats clustering section
        self._update_stats_stage("clustering", {
            "status": StageStatus.COMPLETED.value,
            "method": self.config.clustering_method.value,
            "threshold": self.config.clustering_threshold,
            "num_clusters": len(clusters),
            "cluster_sizes": [len(c) for c in clusters],
            "duration_seconds": round(duration, 2)
        })

        logger.info(f"Updated {self.FILES['clusters']}")

    def update_induction_progress(self, cluster_result: Dict[str, Any]):
        """Append a single cluster's induction result (called after each cluster)."""
        # Read current state
        path = self.output_dir / self.FILES["induction"]
        with open(path, "r") as f:
            data = json.load(f)

        # Append new result
        data["cluster_results"].append(cluster_result)
        data["metadata"]["status"] = StageStatus.IN_PROGRESS.value
        data["metadata"]["num_clusters_processed"] = len(data["cluster_results"])
        data["metadata"]["total_experiences"] = sum(
            r["num_experiences_extracted"] for r in data["cluster_results"]
        )
        data["metadata"]["generated_at"] = datetime.now().isoformat()

        self._write_json(self.FILES["induction"], data)

    def finalize_induction(self, duration: float):
        """Mark induction as complete."""
        path = self.output_dir / self.FILES["induction"]
        with open(path, "r") as f:
            data = json.load(f)

        data["metadata"]["status"] = StageStatus.COMPLETED.value
        data["metadata"]["generated_at"] = datetime.now().isoformat()

        self._write_json(self.FILES["induction"], data)

        # Update pipeline_stats
        experiences_per_cluster = [
            r["num_experiences_extracted"] for r in data["cluster_results"]
        ]
        self._update_stats_stage("induction", {
            "status": StageStatus.COMPLETED.value,
            "level": self.config.workflow_level.name,
            "num_experiences": sum(experiences_per_cluster),
            "experiences_per_cluster": experiences_per_cluster,
            "duration_seconds": round(duration, 2)
        })

        logger.info(f"Finalized {self.FILES['induction']}")

    def finalize_workflows(self, workflows: List[Workflow], num_trajectories: int):
        """Write final workflows.json."""
        self._write_json(self.FILES["workflows"], {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "status": StageStatus.COMPLETED.value,
                "num_trajectories": num_trajectories,
                "config": {
                    "compression": self.config.compression_strategy.value,
                    "clustering": self.config.clustering_method.value,
                    "threshold": self.config.clustering_threshold,
                    "level": self.config.workflow_level.name,
                    "model": self.config.llm_model,
                }
            },
            "workflows": [w.to_dict() for w in workflows],
            "count": len(workflows)
        })
        logger.info(f"Finalized {self.FILES['workflows']}")

    def finalize_summary(self, workflows: List[Workflow], num_trajectories: int):
        """Write final summary.txt."""
        lines = [
            f"CAWM Workflow Extraction Summary",
            f"{'=' * 40}\n",
            f"Status: {StageStatus.COMPLETED.value}",
            f"Generated: {datetime.now().isoformat()}",
            f"Trajectories processed: {num_trajectories}",
            f"Experiences extracted: {len(workflows)}\n",
            f"Configuration:",
            f"  Compression: {self.config.compression_strategy.value}",
            f"  Clustering: {self.config.clustering_method.value} (threshold={self.config.clustering_threshold})",
            f"  Level: {self.config.workflow_level.name}",
            f"  Model: {self.config.llm_model}\n",
            f"Experiences:",
            f"{'-' * 40}",
        ]

        for i, wf in enumerate(workflows, 1):
            lines.append(f"\n{i}. {wf.description}")
            lines.append(f"   Category: {wf.category}")
            lines.append(f"   Steps: {len(wf.steps)}")
            for j, step in enumerate(wf.steps, 1):
                action_preview = step.action[:60] + "..." if len(step.action) > 60 else step.action
                lines.append(f"     {j}. [{step.action_type}] {action_preview}")

        self._write_text(self.FILES["summary"], "\n".join(lines))
        logger.info(f"Finalized {self.FILES['summary']}")

    def finalize_stats(self, total_duration: float):
        """Finalize pipeline_stats.json."""
        path = self.output_dir / self.FILES["stats"]
        with open(path, "r") as f:
            data = json.load(f)

        data["status"] = StageStatus.COMPLETED.value
        data["end_time"] = datetime.now().isoformat()
        data["total_duration_seconds"] = round(total_duration, 2)

        self._write_json(self.FILES["stats"], data)
        logger.info(f"Finalized {self.FILES['stats']}")

    def mark_failed(self, error_message: str):
        """Mark pipeline as failed in all output files."""
        timestamp = datetime.now().isoformat()

        # Update stats
        path = self.output_dir / self.FILES["stats"]
        try:
            with open(path, "r") as f:
                data = json.load(f)
            data["status"] = StageStatus.FAILED.value
            data["end_time"] = timestamp
            data["error"] = error_message
            self._write_json(self.FILES["stats"], data)
        except Exception:
            pass

        # Update summary
        self._write_text(self.FILES["summary"],
            f"CAWM Workflow Extraction Summary\n"
            f"{'=' * 40}\n\n"
            f"Status: {StageStatus.FAILED.value}\n"
            f"Error: {error_message}\n"
            f"Timestamp: {timestamp}\n"
        )

        logger.error(f"Pipeline marked as failed: {error_message}")

    def _update_stats_stage(self, stage: str, stage_data: Dict[str, Any]):
        """Update a specific stage in pipeline_stats.json."""
        path = self.output_dir / self.FILES["stats"]
        with open(path, "r") as f:
            data = json.load(f)

        data["stages"][stage] = stage_data
        data["status"] = StageStatus.IN_PROGRESS.value

        self._write_json(self.FILES["stats"], data)

@dataclass
class PipelineConfig:
    # Compression settings
    compression_strategy: CompressionStrategy = CompressionStrategy.KEY_STEP_EXTRACTION

    # Clustering settings
    # Use problem_description for better semantic grouping
    # Lower threshold (0.2) to create more diverse clusters
    clustering_method: SimilarityMethod = SimilarityMethod.PROBLEM_DESCRIPTION
    clustering_threshold: float = 0.2

    # Induction settings
    workflow_level: WorkflowLevel = WorkflowLevel.GENERAL

    # LLM settings (if not passed explicitly)
    llm_model: str = "moonshotai/kimi-k2-0905"

@dataclass
class PipelineStats:
    """Statistics collected during pipeline execution."""
    start_time: str = ""
    end_time: str = ""
    total_duration_seconds: float = 0.0
    num_input_trajectories: int = 0
    num_clusters: int = 0
    num_experiences: int = 0
    clustering_duration_seconds: float = 0.0
    induction_duration_seconds: float = 0.0
    cluster_sizes: List[int] = field(default_factory=list)
    experiences_per_cluster: List[int] = field(default_factory=list)


class CAWMPipeline:
    """
    Orchestrates the complete CAWM workflow extraction process.
    Pipeline: Load -> Compress -> Cluster -> Induce -> Save

    Uses OutputManager for standardized file output:
    - All output files are created at pipeline start
    - Files are updated progressively as each stage completes
    - Consistent output structure regardless of configuration
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[PipelineConfig] = None,
        output_dir: Optional[str] = None
    ):
        self.llm_client = llm_client
        self.config = config or PipelineConfig()
        self.output_dir = Path(output_dir) if output_dir else None

        # Initialize OutputManager if output_dir is provided
        self.output_manager: Optional[OutputManager] = None
        if self.output_dir:
            self.output_manager = OutputManager(self.output_dir, self.config)

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

        num_trajectories = len(trajectories)
        pipeline_start = time.time()

        # Initialize all output files at the start
        if self.output_manager:
            self.output_manager.initialize_all(num_trajectories)

        logger.info(f"Starting pipeline with {num_trajectories} trajectories")

        try:
            # 1. Cluster
            logger.info("Clustering trajectories...")
            cluster_start = time.time()
            clusters = self.clusterer.cluster(trajectories)
            clustering_duration = time.time() - cluster_start
            logger.info(f"Formed {len(clusters)} clusters")

            # Update clustering output immediately
            if self.output_manager:
                self.output_manager.update_clustering(clusters, clustering_duration)

            # 2. Induce (with progressive updates)
            logger.info("Inducing workflows...")
            induction_start = time.time()
            workflows = self._induce_with_progress(clusters)
            induction_duration = time.time() - induction_start
            logger.info(f"Extracted {len(workflows)} experiences")

            # Finalize induction output
            if self.output_manager:
                self.output_manager.finalize_induction(induction_duration)

            # 3. Finalize all outputs
            total_duration = time.time() - pipeline_start
            if self.output_manager:
                self.output_manager.finalize_workflows(workflows, num_trajectories)
                self.output_manager.finalize_summary(workflows, num_trajectories)
                self.output_manager.finalize_stats(total_duration)

            return workflows

        except Exception as e:
            # Mark pipeline as failed
            if self.output_manager:
                self.output_manager.mark_failed(str(e))
            raise

    def _induce_with_progress(self, clusters: List[TrajectoryCluster]) -> List[Workflow]:
        """Induce workflows with progressive file updates after each cluster."""
        all_workflows = []

        for cluster in clusters:
            cluster_start = time.time()
            logger.info(f"Inducing from cluster {cluster.cluster_id} ({len(cluster)} trajectories)")

            # Get workflows for this cluster
            wfs = self.inductor.induce(cluster.trajectories, self.config.workflow_level)

            # Build cluster result
            cluster_result = {
                "cluster_id": cluster.cluster_id,
                "cluster_label": cluster.label,
                "num_trajectories": len(cluster),
                "trajectory_ids": [t.instance_id for t in cluster.trajectories],
                "num_experiences_extracted": len(wfs),
                "experiences": [
                    {
                        "id": wf.id,
                        "description": wf.description,
                        "category": wf.category,
                        "trigger": wf.steps[0].env_description if wf.steps else "",
                        "insight": wf.steps[0].reasoning if wf.steps else "",
                        "action": wf.steps[0].action if wf.steps else "",
                    }
                    for wf in wfs
                ],
                "duration_seconds": round(time.time() - cluster_start, 2)
            }

            # Update induction file immediately after each cluster
            if self.output_manager:
                self.output_manager.update_induction_progress(cluster_result)

            all_workflows.extend(wfs)

        return all_workflows

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
