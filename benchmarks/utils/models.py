from typing import Any

from pydantic import BaseModel, Field, model_validator

from openhands.sdk import LLM, get_logger


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
    critic_name: str | None = Field(
        default=None,
        description=(
            "Name of the critic to use for evaluation (required unless max_attempts=1)"
        ),
    )

    @model_validator(mode="after")
    def validate_critic_name(self):
        if self.max_attempts != 1 and self.critic_name is None:
            raise ValueError("critic_name is required when max_attempts is not 1")
        return self


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
    history: list[Any] | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None

    # Optionally save the input test instance
    instance: dict[str, Any] | None = None
