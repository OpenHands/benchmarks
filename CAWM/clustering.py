from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import random
import re
from io import StringIO

try:
    from unidiff import PatchSet
    HAS_UNIDIFF = True
except ImportError:
    HAS_UNIDIFF = False

from CAWM.models import Trajectory, TrajectoryCluster, ActionType
from CAWM.llm_client import LLMClient

logger = logging.getLogger(__name__)

class SimilarityMethod(Enum):
    PROBLEM_DESCRIPTION = "problem_description"
    ACTION_SEQUENCE = "action_sequence"
    CODE_MODIFICATION = "code_modification"
    RANDOM = "random"  # For baseline/testing

@dataclass
class ClusteringConfig:
    method: SimilarityMethod = SimilarityMethod.ACTION_SEQUENCE
    n_clusters: int = 5
    threshold: float = 0.5
    
class ClusteringModule:
    """
    Module for clustering similar trajectories.
    """
    def __init__(
        self, 
        method: Union[str, SimilarityMethod] = SimilarityMethod.ACTION_SEQUENCE,
        llm_client: Optional[LLMClient] = None,
        config: Optional[ClusteringConfig] = None
    ):
        if isinstance(method, str):
            try:
                self.method = SimilarityMethod(method)
            except ValueError:
                logger.warning(f"Unknown method {method}, falling back to ACTION_SEQUENCE")
                self.method = SimilarityMethod.ACTION_SEQUENCE
        else:
            self.method = method
            
        self.llm_client = llm_client
        self.config = config or ClusteringConfig(method=self.method)
        self.config.method = self.method

    def cluster(self, trajectories: List[Trajectory]) -> List[TrajectoryCluster]:
        """Cluster trajectories based on configured method."""
        if not trajectories:
            return []
            
        if self.method == SimilarityMethod.RANDOM:
            return self._cluster_random(trajectories)
            
        elif self.method == SimilarityMethod.ACTION_SEQUENCE:
            return self._cluster_action_sequence(trajectories)
            
        elif self.method == SimilarityMethod.PROBLEM_DESCRIPTION:
            return self._cluster_problem_description(trajectories)
            
        elif self.method == SimilarityMethod.CODE_MODIFICATION:
            return self._cluster_code_modification(trajectories)
            
        return [TrajectoryCluster(cluster_id="all", trajectories=trajectories, label="All")]

    def get_similarity(self, traj1: Trajectory, traj2: Trajectory) -> float:
        """Calculate similarity between two trajectories."""
        if self.method == SimilarityMethod.ACTION_SEQUENCE:
            return self._calc_seq_similarity(traj1, traj2)
        elif self.method == SimilarityMethod.PROBLEM_DESCRIPTION:
            return self._calc_text_similarity(traj1.instruction, traj2.instruction)
        elif self.method == SimilarityMethod.CODE_MODIFICATION:
            return self._calc_code_similarity(traj1, traj2)
        return 0.0

    def _cluster_random(self, trajectories: List[Trajectory]) -> List[TrajectoryCluster]:
        """Random clustering for testing."""
        clusters = {}
        n = self.config.n_clusters
        
        for traj in trajectories:
            cid = f"cluster_{random.randint(0, n-1)}"
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(traj)
            
        return [
            TrajectoryCluster(
                cluster_id=cid, 
                trajectories=ts, 
                label=f"Random Cluster {cid}",
                similarity_method=self.method.value
            ) 
            for cid, ts in clusters.items()
        ]

    def _cluster_action_sequence(self, trajectories: List[Trajectory]) -> List[TrajectoryCluster]:
        """
        Cluster based on Jaccard similarity of action type n-grams or simpler sequence matching.
        """
        # Simple greedy clustering
        clusters: List[TrajectoryCluster] = []
        unassigned = trajectories.copy()
        
        cluster_idx = 0
        while unassigned:
            seed = unassigned.pop(0)
            current_cluster = [seed]
            
            # Find similar
            remaining = []
            for candidate in unassigned:
                sim = self._calc_seq_similarity(seed, candidate)
                if sim >= self.config.threshold:
                    current_cluster.append(candidate)
                else:
                    remaining.append(candidate)
            
            unassigned = remaining
            
            clusters.append(TrajectoryCluster(
                cluster_id=f"seq_cluster_{cluster_idx}",
                trajectories=current_cluster,
                label=f"Action Sequence Cluster {cluster_idx}",
                similarity_method=self.method.value
            ))
            cluster_idx += 1
            
        return clusters

    def _calc_seq_similarity(self, t1: Trajectory, t2: Trajectory) -> float:
        """Jaccard similarity of action types."""
        s1 = set(t1.get_action_sequence())
        s2 = set(t2.get_action_sequence())
        
        if not s1 or not s2:
            return 0.0
            
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        return intersection / union

    def _cluster_problem_description(self, trajectories: List[Trajectory]) -> List[TrajectoryCluster]:
        """Cluster based on text similarity of problem description."""
        clusters: List[TrajectoryCluster] = []
        unassigned = trajectories.copy()
        cluster_idx = 0
        
        while unassigned:
            seed = unassigned.pop(0)
            current_cluster = [seed]
            remaining = []
            
            for candidate in unassigned:
                sim = self._calc_text_similarity(seed.instruction, candidate.instruction)
                if sim >= self.config.threshold:
                    current_cluster.append(candidate)
                else:
                    remaining.append(candidate)
            
            unassigned = remaining
            clusters.append(TrajectoryCluster(
                cluster_id=f"text_cluster_{cluster_idx}",
                trajectories=current_cluster,
                label=f"Problem Desc Cluster {cluster_idx}",
                similarity_method=self.method.value
            ))
            cluster_idx += 1
            
        return clusters

    def _calc_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity of tokens."""
        def tokenize(t):
            return set(re.findall(r'\w+', t.lower()))
            
        s1 = tokenize(text1)
        s2 = tokenize(text2)
        
        if not s1 or not s2:
            return 0.0
            
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        return intersection / union

    def _cluster_code_modification(self, trajectories: List[Trajectory]) -> List[TrajectoryCluster]:
        """Cluster based on modified files similarity."""
        if not HAS_UNIDIFF:
            logger.warning("unidiff not available, skipping code modification clustering")
            return [TrajectoryCluster("default", trajectories, "Default")]

        clusters: List[TrajectoryCluster] = []
        unassigned = trajectories.copy()
        cluster_idx = 0
        
        while unassigned:
            seed = unassigned.pop(0)
            current_cluster = [seed]
            remaining = []
            
            for candidate in unassigned:
                sim = self._calc_code_similarity(seed, candidate)
                if sim >= self.config.threshold:
                    current_cluster.append(candidate)
                else:
                    remaining.append(candidate)
            
            unassigned = remaining
            clusters.append(TrajectoryCluster(
                cluster_id=f"code_cluster_{cluster_idx}",
                trajectories=current_cluster,
                label=f"Code Mod Cluster {cluster_idx}",
                similarity_method=self.method.value
            ))
            cluster_idx += 1
            
        return clusters

    def _extract_modified_files(self, patch_str: str) -> set:
        if not patch_str:
            return set()
        try:
            patch = PatchSet(StringIO(patch_str))
            return {f.path for f in patch}
        except Exception:
            return set()

    def _calc_code_similarity(self, t1: Trajectory, t2: Trajectory) -> float:
        s1 = self._extract_modified_files(t1.git_patch or "")
        s2 = self._extract_modified_files(t2.git_patch or "")
        
        if not s1 or not s2:
            return 0.0
            
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        return intersection / union