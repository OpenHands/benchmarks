import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from openhands.sdk import LLM, Event, get_logger
from openhands.sdk.critic import CriticBase
from openhands.sdk.llm import Metrics
from openhands.sdk.llm.utils.metrics import TokenUsage
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


class CostBreakdown(BaseModel):
    """Normalized cost structure attached to each attempt."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    input_cost: float | None = Field(default=None, ge=0)
    output_cost: float | None = Field(default=None, ge=0)
    total_cost: float | None = Field(default=None, ge=0)


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
    # output of the evaluation
    # store anything that is needed for the score calculation
    test_result: dict[str, Any]

    instruction: str | None = None

    # Interaction info
    metadata: EvalMetadata | None = None
    history: list[Event] = Field(default_factory=list)
    metrics: Metrics | None = None
    error: str | None = None
    # Outcome details
    status: Literal["success", "error"] | None = None
    resolved: bool | None = None
    attempt: int | None = None
    max_attempts: int | None = None
    duration_seconds: float | None = None
    cost: CostBreakdown | None = None
    artifacts_url: str | None = None

    # Optionally save the input test instance
    instance: dict[str, Any] | None = None
    artifacts: dict[str, Any] | None = None

    def _require_output_fields(self) -> None:
        missing = []
        if self.attempt is None:
            missing.append("attempt")
        if self.max_attempts is None:
            missing.append("max_attempts")
        if self.status is None:
            missing.append("status")
        if self.cost is None:
            missing.append("cost")
        if self.artifacts_url is None:
            missing.append("artifacts_url")
        if missing:
            raise ValueError(
                "EvalOutput missing required fields for output.jsonl: "
                + ", ".join(missing)
            )

    def to_output_dict(self) -> dict[str, Any]:
        """Return the canonical output.jsonl representation."""
        self._require_output_fields()
        payload = {
            "instance_id": self.instance_id,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "status": self.status,
            "resolved": self.resolved,
            "error": self.error,
            "test_result": self.test_result or {},
            "cost": self.cost.model_dump(),
            "artifacts_url": self.artifacts_url,
        }
        if self.duration_seconds is not None:
            payload["duration_seconds"] = self.duration_seconds
        return payload


_CANONICAL_FIELDS = {
    "instance_id",
    "attempt",
    "max_attempts",
    "status",
    "resolved",
    "error",
    "test_result",
    "duration_seconds",
    "cost",
    "artifacts_url",
}
_CANONICAL_REQUIRED_FIELDS = {
    "instance_id",
    "attempt",
    "max_attempts",
    "status",
    "resolved",
    "error",
    "test_result",
    "cost",
    "artifacts_url",
}
_CANONICAL_COST_FIELDS = {
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_cost",
    "output_cost",
    "total_cost",
}


def _tokens_from_usage(token_usage: TokenUsage | None) -> tuple[int | None, int | None]:
    """Extract prompt/completion tokens safely."""
    if token_usage is None:
        return None, None
    return token_usage.prompt_tokens, token_usage.completion_tokens


def cost_from_metrics(metrics: Metrics | None) -> CostBreakdown:
    """Map SDK metrics to the standardized cost block."""
    if metrics is None:
        return CostBreakdown()

    prompt_tokens, completion_tokens = _tokens_from_usage(
        metrics.accumulated_token_usage
    )
    total_tokens = (
        prompt_tokens + completion_tokens
        if prompt_tokens is not None and completion_tokens is not None
        else None
    )

    return CostBreakdown(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
        # The SDK tracks only total cost; leave per-direction costs unset.
        total_cost=metrics.accumulated_cost,
    )


def _validate_canonical_output_dict(data: dict[str, Any]) -> None:
    extra = set(data) - _CANONICAL_FIELDS
    if extra:
        raise ValueError(f"Unexpected keys in output.jsonl: {sorted(extra)}")

    missing = _CANONICAL_REQUIRED_FIELDS - set(data)
    if missing:
        raise ValueError(f"Missing required keys in output.jsonl: {sorted(missing)}")

    if not isinstance(data["instance_id"], str):
        raise ValueError("instance_id must be a string")
    if not isinstance(data["attempt"], int) or data["attempt"] < 1:
        raise ValueError("attempt must be an int >= 1")
    if not isinstance(data["max_attempts"], int) or data["max_attempts"] < 1:
        raise ValueError("max_attempts must be an int >= 1")
    if data["status"] not in ("success", "error"):
        raise ValueError("status must be 'success' or 'error'")
    if data["resolved"] is not None and not isinstance(data["resolved"], bool):
        raise ValueError("resolved must be a bool or null")
    if "duration_seconds" in data and data["duration_seconds"] is not None:
        duration = data["duration_seconds"]
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or duration < 0
        ):
            raise ValueError("duration_seconds must be a non-negative number or null")
    if not isinstance(data["test_result"], dict):
        raise ValueError("test_result must be an object")
    if not isinstance(data["cost"], dict):
        raise ValueError("cost must be an object")
    extra_cost = set(data["cost"]) - _CANONICAL_COST_FIELDS
    if extra_cost:
        raise ValueError(f"Unexpected keys in cost: {sorted(extra_cost)}")
    if not isinstance(data["artifacts_url"], str):
        raise ValueError("artifacts_url must be a string")

    CostBreakdown.model_validate(data["cost"])


def write_output_line(output_path: Path, output: EvalOutput) -> None:
    """Append one canonical line to output.jsonl with a file lock."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = output.to_output_dict()
    _validate_canonical_output_dict(payload)
    with open(output_path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(payload) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def load_output_file(path: str | Path) -> list[EvalOutput]:
    """Load a canonical output.jsonl into typed objects (strict validation)."""
    outputs: list[EvalOutput] = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Output file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                _validate_canonical_output_dict(data)
                outputs.append(EvalOutput.model_validate(data))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} in {path}: {exc}") from exc
            except ValueError as exc:
                raise ValueError(f"Invalid line {line_num} in {path}: {exc}") from exc
    return outputs


def validate_output_file(path: str | Path) -> None:
    """Validate that every line in the file conforms to the schema."""
    load_output_file(path)


def select_best_attempts(
    outputs: Sequence[EvalOutput],
) -> dict[str, EvalOutput]:
    """
    Collapse attempts per instance, preferring successful attempts.

    Picks the highest-attempt successful line if present,
    otherwise the highest-attempt line.
    """
    best: dict[str, EvalOutput] = {}
    for out in sorted(outputs, key=lambda o: (o.instance_id, o.attempt or 0)):
        existing = best.get(out.instance_id)
        if existing is None:
            best[out.instance_id] = out
            continue

        if out.resolved:
            if not existing.resolved or (out.attempt or 0) >= (existing.attempt or 0):
                best[out.instance_id] = out
            continue

        if (out.attempt or 0) >= (existing.attempt or 0) and not existing.resolved:
            best[out.instance_id] = out
    return best


def derive_report(outputs: Iterable[EvalOutput]) -> dict[str, Any]:
    """Generate a derived summary report from canonical outputs."""
    outputs = list(outputs)
    best = select_best_attempts(outputs)
    resolved_ids = [k for k, v in best.items() if v.resolved]
    unresolved_ids = [k for k, v in best.items() if v.resolved is False]
    error_ids = [k for k, v in best.items() if v.status == "error"]
    agent_failures = [k for k in unresolved_ids if k not in error_ids]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "derived_from": "output.jsonl",
        "totals": {
            "attempt_lines": len(outputs),
            "instances": len(best),
            "resolved": len(resolved_ids),
            "agent_failures": len(agent_failures),
            "errors": len(error_ids),
            "unresolved": len(unresolved_ids),
        },
        "resolved_ids": resolved_ids,
        "agent_failure_ids": agent_failures,
        "error_ids": error_ids,
        "unresolved_ids": unresolved_ids,
    }


def write_derived_report(output_dir: str | Path) -> Path:
    """Create output.report.json next to output.jsonl."""
    output_dir = Path(output_dir)
    output_file = output_dir / "output.jsonl"
    report_file = output_dir / "output.report.json"

    outputs = load_output_file(output_file)
    report = derive_report(outputs)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report_file
