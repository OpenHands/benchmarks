from typing import Any

from pydantic import BaseModel, Field

from openhands.sdk import LLM, Agent, get_logger


logger = get_logger(__name__)


class EvalMetadata(BaseModel):
    llm: LLM
    dataset: str
    max_iterations: int
    eval_output_dir: str
    data_split: str | None = None
    details: dict[str, Any] | None = None
    prompt_path: str | None = None
    eval_n_limit: int | None = None
    env_setup_commands: list[str] | None = None

EvalInstanceID = str

class EvalInstance(BaseModel):
    """
    Represents a single evaluation instance.

    This class provides a structured way to represent instances across different
    benchmarks while maintaining flexibility through the generic data field.
    """

    id: EvalInstanceID = Field(..., description="Mandatory unique identifier")
    data: dict[str, Any] = Field(..., description="Generic data field for benchmark-specific content")


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
