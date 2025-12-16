from typing import Any, Literal

from pydantic import BaseModel, Field

from openhands.sdk import LLM, Event, get_logger
from openhands.sdk.critic import CriticBase
from openhands.sdk.llm import Metrics


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
    instance_ids: str | None = Field(
        default=None,
        description=(
            "Comma-separated list of instance IDs to evaluate "
            "(e.g., 'id1,id2,id3'). Takes precedence over selected_instances_file."
        ),
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


class EvalOutput(BaseModel):
    # NOTE: User-specified
    instance_id: str
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
