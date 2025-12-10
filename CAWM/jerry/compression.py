import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

from CAWM.llm_client import LLMClient
from CAWM.models import ActionType, Trajectory, TrajectoryEvent


logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    KEY_STEP_EXTRACTION = "key_step_extraction"
    HIERARCHICAL_SUMMARIZATION = "hierarchical_summarization"
    ACTION_TYPE_FILTERING = "action_type_filtering"
    NO_OP = "no_op"


@dataclass
class CompressionConfig:
    strategy: CompressionStrategy = CompressionStrategy.KEY_STEP_EXTRACTION
    # For Filtering
    keep_action_types: List[ActionType] = field(
        default_factory=lambda: [
            ActionType.FILE_EDIT,
            ActionType.TESTING,
            ActionType.EXPLORATION,
        ]
    )
    # For Summarization
    chunk_size: int = 10
    summary_prompt_template: str = "Summarize these actions into high-level steps:"


class CompressionModule:
    """
    Module for compressing trajectories to reduce length while retaining key info.
    """

    def __init__(
        self,
        strategy: Union[
            str, CompressionStrategy
        ] = CompressionStrategy.KEY_STEP_EXTRACTION,
        llm_client: Optional[LLMClient] = None,
        config: Optional[CompressionConfig] = None,
    ):
        if isinstance(strategy, str):
            try:
                self.strategy = CompressionStrategy(strategy)
            except ValueError:
                logger.warning(
                    f"Unknown strategy {strategy}, falling back to KEY_STEP_EXTRACTION"
                )
                self.strategy = CompressionStrategy.KEY_STEP_EXTRACTION
        else:
            self.strategy = strategy

        self.llm_client = llm_client
        self.config = config or CompressionConfig(strategy=self.strategy)

        # Update config strategy if passed in constructor
        self.config.strategy = self.strategy

    def compress(self, trajectory: Trajectory) -> Trajectory:
        """Compress a single trajectory."""
        if self.strategy == CompressionStrategy.NO_OP:
            return trajectory

        if self.strategy == CompressionStrategy.KEY_STEP_EXTRACTION:
            return self._compress_key_extraction(trajectory)

        elif self.strategy == CompressionStrategy.ACTION_TYPE_FILTERING:
            return self._compress_filtering(trajectory)

        elif self.strategy == CompressionStrategy.HIERARCHICAL_SUMMARIZATION:
            return self._compress_summarization(trajectory)

        return trajectory

    def compress_batch(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """Compress a batch of trajectories."""
        return [self.compress(t) for t in trajectories]

    def _compress_key_extraction(self, trajectory: Trajectory) -> Trajectory:
        """
        Strategy 1: Keep only key steps (edits, tests) and context around them.
        """
        if not trajectory.events:
            return trajectory

        key_indices = [i for i, e in enumerate(trajectory.events) if e.is_key_step()]
        if not key_indices:
            # If no key steps, keep start and end
            indices_to_keep = {0, len(trajectory.events) - 1}
        else:
            indices_to_keep = set()
            for idx in key_indices:
                # Keep 1 step before and after for context
                indices_to_keep.add(max(0, idx - 1))
                indices_to_keep.add(idx)
                indices_to_keep.add(min(len(trajectory.events) - 1, idx + 1))

        new_events = [trajectory.events[i] for i in sorted(list(indices_to_keep))]

        # Create new trajectory with subset of events
        new_traj = Trajectory(
            instance_id=trajectory.instance_id,
            instruction=trajectory.instruction,
            events=new_events,
            git_patch=trajectory.git_patch,
            repository=trajectory.repository,
            issue_type=trajectory.issue_type,
            metadata=trajectory.metadata.copy(),
        )
        new_traj.metadata["compression"] = "key_step_extraction"
        return new_traj

    def _compress_filtering(self, trajectory: Trajectory) -> Trajectory:
        """
        Strategy 3: Filter by ActionType.
        """
        allowed = set(self.config.keep_action_types)
        new_events = [e for e in trajectory.events if e.action_type in allowed]

        new_traj = Trajectory(
            instance_id=trajectory.instance_id,
            instruction=trajectory.instruction,
            events=new_events,
            git_patch=trajectory.git_patch,
            repository=trajectory.repository,
            issue_type=trajectory.issue_type,
            metadata=trajectory.metadata.copy(),
        )
        new_traj.metadata["compression"] = "action_type_filtering"
        return new_traj

    def _compress_summarization(self, trajectory: Trajectory) -> Trajectory:
        """
        Strategy 2: LLM-based summarization.
        """
        if not self.llm_client:
            logger.warning("LLMClient required for summarization. Returning original.")
            return trajectory

        chunk_size = self.config.chunk_size
        if not trajectory.events:
            return trajectory

        # Split into chunks
        events = trajectory.events
        chunks = [events[i : i + chunk_size] for i in range(0, len(events), chunk_size)]

        new_events = []

        for i, chunk in enumerate(chunks):
            summary_text = self._generate_chunk_summary(chunk)

            first_event = chunk[0]

            # Create a pseudo-event for the summary
            # Use 'THINK' action type as it represents high level reasoning
            summary_event = TrajectoryEvent(
                index=first_event.index,
                kind="ActionEvent",
                action_type=ActionType.THINK,
                action={"kind": "ThinkAction", "thought": summary_text},
                action_kind="ThinkAction",
                thought=[summary_text],
                command=None,
                path=None,
            )
            new_events.append(summary_event)

        new_traj = Trajectory(
            instance_id=trajectory.instance_id,
            instruction=trajectory.instruction,
            events=new_events,
            git_patch=trajectory.git_patch,
            repository=trajectory.repository,
            issue_type=trajectory.issue_type,
            metadata=trajectory.metadata.copy(),
        )
        new_traj.metadata["compression"] = "hierarchical_summarization"
        return new_traj

    def _generate_chunk_summary(self, chunk: List[TrajectoryEvent]) -> str:
        lines = []
        for e in chunk:
            lines.append(f"- [{e.action_type.value}] {e.get_action_summary()}")

        text = "\n".join(lines)
        prompt = f"{self.config.summary_prompt_template}\n\n{text}"

        return self.llm_client.complete(prompt, system="You are a concise summarizer.")

    def __add__(self, other: "CompressionModule") -> "ComposedCompressionModule":
        """Support composition: module1 + module2"""
        return ComposedCompressionModule([self, other])


class ComposedCompressionModule(CompressionModule):
    """
    Executes multiple compression modules in sequence.
    """

    def __init__(self, modules: List[CompressionModule]):
        self.modules = modules
        # Config is ambiguous for composed, so None
        super().__init__(strategy=CompressionStrategy.NO_OP)

    def compress(self, trajectory: Trajectory) -> Trajectory:
        current = trajectory
        for module in self.modules:
            current = module.compress(current)
        return current

    def __add__(self, other: CompressionModule) -> "ComposedCompressionModule":
        if isinstance(other, ComposedCompressionModule):
            return ComposedCompressionModule(self.modules + other.modules)
        return ComposedCompressionModule(self.modules + [other])
