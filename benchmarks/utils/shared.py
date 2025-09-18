from typing import Any

from pydantic import BaseModel

from openhands.sdk import LLM, Agent, Event, get_logger


logger = get_logger(__name__)


class EvalMetadata(BaseModel):
    llm: LLM
    agent_config: Agent | None = None
    max_iterations: int
    eval_output_dir: str
    dataset: str | None = None
    data_split: str | None = None
    details: dict[str, Any] | None = None
    # TODO: calvin is porting this over as a pydantic class
    # condenser_config: Condenser | None = None
    instruction_template_name: str | None = None
    # New fields for refactoring
    prompt_path: str | None = None
    eval_n_limit: int | None = None
    env_setup_commands: list[str] | None = None


class EvalOutput(BaseModel):
    # NOTE: User-specified
    instance_id: str
    # output of the evaluation
    # store anything that is needed for the score calculation
    test_result: dict[str, Any]

    instruction: str | None = None

    # Interaction info
    metadata: EvalMetadata | None = None
    history: list[Event] | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None

    # Optionally save the input test instance
    instance: dict[str, Any] | None = None


class EvalException(Exception):
    pass


class EvalTimeoutException(Exception):
    pass
