import logging
import random
import re
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from typing import Dict, List, Optional, Union


try:
    from unidiff import PatchSet

    HAS_UNIDIFF = True
except ImportError:
    HAS_UNIDIFF = False

from CAWM.llm_client import LLMClient
from CAWM.models import Trajectory, TrajectoryCluster


logger = logging.getLogger(__name__)


class SimilarityMethod(Enum):
    PROBLEM_DESCRIPTION = "problem_description"
    ACTION_SEQUENCE = "action_sequence"
    CODE_MODIFICATION = "code_modification"
    REPOSITORY = "repository"  # Group by repository (astropy, django, etc.)
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
        config: Optional[ClusteringConfig] = None,
    ):
        if isinstance(method, str):
            try:
                self.method = SimilarityMethod(method)
            except ValueError:
                logger.warning(
                    f"Unknown method {method}, falling back to ACTION_SEQUENCE"
                )
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

        elif self.method == SimilarityMethod.REPOSITORY:
            return self._cluster_by_repository(trajectories)

        return [
            TrajectoryCluster(cluster_id="all", trajectories=trajectories, label="All")
        ]

    def get_similarity(self, traj1: Trajectory, traj2: Trajectory) -> float:
        """Calculate similarity between two trajectories."""
        if self.method == SimilarityMethod.ACTION_SEQUENCE:
            return self._calc_seq_similarity(traj1, traj2)
        elif self.method == SimilarityMethod.PROBLEM_DESCRIPTION:
            return self._calc_text_similarity(traj1.instruction, traj2.instruction)
        elif self.method == SimilarityMethod.CODE_MODIFICATION:
            return self._calc_code_similarity(traj1, traj2)
        return 0.0

    def _cluster_random(
        self, trajectories: List[Trajectory]
    ) -> List[TrajectoryCluster]:
        """Random clustering for testing."""
        clusters = {}
        n = self.config.n_clusters

        for traj in trajectories:
            cid = f"cluster_{random.randint(0, n - 1)}"
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(traj)

        return [
            TrajectoryCluster(
                cluster_id=cid,
                trajectories=ts,
                label=f"Random Cluster {cid}",
                similarity_method=self.method.value,
            )
            for cid, ts in clusters.items()
        ]

    def _cluster_action_sequence(
        self, trajectories: List[Trajectory]
    ) -> List[TrajectoryCluster]:
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

            clusters.append(
                TrajectoryCluster(
                    cluster_id=f"seq_cluster_{cluster_idx}",
                    trajectories=current_cluster,
                    label=f"Action Sequence Cluster {cluster_idx}",
                    similarity_method=self.method.value,
                )
            )
            cluster_idx += 1

        return clusters

    def _calc_seq_similarity(self, t1: Trajectory, t2: Trajectory) -> float:
        """
        Calculate sequence similarity using n-gram Jaccard similarity.

        Instead of just using set of action types (which loses order),
        we use n-grams (bigrams and trigrams) to capture sequential patterns.
        This ensures trajectories with similar patterns cluster together.
        """
        seq1 = [a.name for a in t1.get_action_sequence()]
        seq2 = [a.name for a in t2.get_action_sequence()]

        if not seq1 or not seq2:
            return 0.0

        # Generate n-grams (bigrams and trigrams)
        def get_ngrams(seq: list, n: int) -> set:
            if len(seq) < n:
                return set()
            return {tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)}

        # Use combination of bigrams and trigrams for pattern matching
        bigrams1 = get_ngrams(seq1, 2)
        bigrams2 = get_ngrams(seq2, 2)
        trigrams1 = get_ngrams(seq1, 3)
        trigrams2 = get_ngrams(seq2, 3)

        # Combine all n-grams
        ngrams1 = bigrams1 | trigrams1
        ngrams2 = bigrams2 | trigrams2

        if not ngrams1 or not ngrams2:
            # Fallback to simple set similarity for very short sequences
            s1 = set(seq1)
            s2 = set(seq2)
            if not s1 or not s2:
                return 0.0
            return len(s1.intersection(s2)) / len(s1.union(s2))

        # Jaccard similarity of n-grams
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        return intersection / union

    def _cluster_problem_description(
        self, trajectories: List[Trajectory]
    ) -> List[TrajectoryCluster]:
        """Cluster based on text similarity of problem description."""
        clusters: List[TrajectoryCluster] = []
        unassigned = trajectories.copy()
        cluster_idx = 0

        while unassigned:
            seed = unassigned.pop(0)
            current_cluster = [seed]
            remaining = []

            for candidate in unassigned:
                sim = self._calc_text_similarity(
                    seed.instruction, candidate.instruction
                )
                if sim >= self.config.threshold:
                    current_cluster.append(candidate)
                else:
                    remaining.append(candidate)

            unassigned = remaining
            clusters.append(
                TrajectoryCluster(
                    cluster_id=f"text_cluster_{cluster_idx}",
                    trajectories=current_cluster,
                    label=f"Problem Desc Cluster {cluster_idx}",
                    similarity_method=self.method.value,
                )
            )
            cluster_idx += 1

        return clusters

    def _cluster_by_repository(
        self, trajectories: List[Trajectory]
    ) -> List[TrajectoryCluster]:
        """
        Cluster trajectories by repository (e.g., astropy, django, sympy).

        This is useful for extracting domain-specific experiences since bugs
        in the same repository often share similar patterns and solutions.
        """
        repo_groups: Dict[str, List[Trajectory]] = {}

        for traj in trajectories:
            # Extract repository from instance_id (e.g., "django__django-12345" -> "django")
            repo = traj.repository or "unknown"
            if repo not in repo_groups:
                repo_groups[repo] = []
            repo_groups[repo].append(traj)

        clusters = []
        for repo, trajs in sorted(repo_groups.items()):
            clusters.append(
                TrajectoryCluster(
                    cluster_id=f"repo_{repo}",
                    trajectories=trajs,
                    label=f"Repository: {repo}",
                    similarity_method=self.method.value,
                )
            )

        return clusters

    def _calc_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity of tokens."""

        def tokenize(t):
            return set(re.findall(r"\w+", t.lower()))

        s1 = tokenize(text1)
        s2 = tokenize(text2)

        if not s1 or not s2:
            return 0.0

        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        return intersection / union

    def _cluster_code_modification(
        self, trajectories: List[Trajectory]
    ) -> List[TrajectoryCluster]:
        """Cluster based on modified files similarity."""
        if not HAS_UNIDIFF:
            logger.warning(
                "unidiff not available, skipping code modification clustering"
            )
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
            clusters.append(
                TrajectoryCluster(
                    cluster_id=f"code_cluster_{cluster_idx}",
                    trajectories=current_cluster,
                    label=f"Code Mod Cluster {cluster_idx}",
                    similarity_method=self.method.value,
                )
            )
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
