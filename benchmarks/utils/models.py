from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from benchmarks.utils.laminar import LaminarEvalMetadata
from openhands.sdk import LLM, Event, get_logger
from openhands.sdk.critic import CriticBase
from openhands.sdk.llm import Metrics
from openhands.sdk.utils.models import OpenHandsModel


logger = get_logger(__name__)


class EvalMetadata(BaseModel):
    llm: LLM
    dataset: str
    dataset_split: str = Field(default="test")
    max_iterations: int
    eval_output_dir: str
    details: dict[str, Any] | None = None
    prompt_path: str | None = Field(
        default=None, description="Path to the prompt template file"
    )
    env_setup_commands: list[str] | None = None
    eval_limit: int = Field(
        default=0, description="Number of instances to evaluate, 0 means all"
    )
    max_attempts: int = Field(
        default=1, ge=1, description="Maximum number of attempts for iterative mode"
    )
    critic: CriticBase = Field(
        description=(
            "Critic instance to use for evaluation. "
            "Critics determine whether an agent's output is considered successful "
            "and whether another attempt should be made in iterative evaluation mode. "
            "If None, a PassCritic will be used (always accepts the output)."
        ),
    )
    selected_instances_file: str | None = Field(
        default=None,
        description="Path to text file containing instance IDs to select "
        "(one per line)",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for instances that throw exceptions",
    )
    workspace_type: Literal["docker", "remote"] = Field(
        default="docker",
        description="Type of workspace to use, e.g., 'docker' or 'remote'",
    )
    base_resource_factor: int = Field(
        default=1,
        ge=1,
        le=8,
        description=(
            "Base resource factor for runtime allocation. "
            "When a runtime crashes, this will be exponentially increased "
            "(2^runtime_failure_count) up to max_resource_factor."
        ),
    )
    max_resource_factor: int = Field(
        default=8,
        ge=1,
        le=16,
        description="Maximum resource factor to use after retries.",
    )
    enable_delegation: bool = Field(
        default=False,
        description="Enable sub-agent delegation tools for the agent",
    )
    lmnr: LaminarEvalMetadata | None = Field(
        default=None,
        description="Laminar evaluation metadata",
    )


EvalInstanceID = str


class EvalInstance(BaseModel):
    """
    Represents a single evaluation instance.

    This class provides a structured way to represent instances across different
    benchmarks while maintaining flexibility through the generic data field.
    """

    id: EvalInstanceID = Field(..., description="Mandatory unique identifier")
    data: dict[str, Any] = Field(
        ..., description="Generic data field for benchmark-specific content"
    )


class RemoteRuntimeAllocation(BaseModel):
    """Mapping of instance → remote runtime (pod) for correlation/debugging.

    Only populated for APIRemoteWorkspace allocations to capture the pod/runtime_id
    used for an attempt/retry so logs can be tied back to instance runs even after
    the pod is gone. Requires at least a runtime_id or session_id to avoid
    meaningless records.
    """

    runtime_id: str | None = Field(
        default=None, description="Runtime/pod identifier from the runtime API"
    )
    session_id: str | None = Field(
        default=None, description="Session identifier used when creating the runtime"
    )
    runtime_url: str | None = Field(
        default=None, description="Base URL for the runtime, if available"
    )
    resource_factor: int | None = Field(
        default=None, description="Resource factor requested for the runtime"
    )
    critic_attempt: int | None = Field(
        default=None, description="Outer critic attempt (1-indexed)"
    )
    retry: int | None = Field(
        default=None, description="Inner retry within the critic attempt (1-indexed)"
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the runtime was allocated",
    )

    @model_validator(mode="after")
    def _require_identifier(self):
        if not self.runtime_id and not self.session_id:
            raise ValueError("runtime_id or session_id is required for remote runtime")
        return self


class EvalOutput(OpenHandsModel):
    """
    Evaluation output model.

    Uses OpenHandsModel to ensure pydantic schemas are properly rebuilt when
    new discriminated union types (like Browser actions/observations) are registered.
    This prevents deserialization errors when loading results that contain
    dynamically registered event types.
    """

    # NOTE: User-specified
    instance_id: str
    attempt: int = Field(
        default=1,
        ge=1,
        description="Attempt number for iterative runs (1-indexed)",
    )
    # output of the evaluation
    # store anything that is needed for the score calculation
    test_result: dict[str, Any]

    instruction: str | None = None

    # Interaction info
    metadata: EvalMetadata | None = None
    history: list[Event] = Field(default_factory=list)
    metrics: Metrics | None = None
    error: str | None = None

    # Optionally save the input test instance
    instance: dict[str, Any] | None = None
    runtime_runs: list[RemoteRuntimeAllocation] | None = Field(
        default=None,
        description=(
            "Remote runtime allocations (pod/session mapping) for this instance"
        ),
    )
