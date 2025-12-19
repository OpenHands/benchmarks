from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from openhands.sdk.llm.utils.metrics import Metrics, TokenUsage


class CostBreakdown(BaseModel):
    """Normalized cost structure attached to each attempt."""

    model_config = ConfigDict(extra="ignore")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    input_cost: float | None = Field(default=None, ge=0)
    output_cost: float | None = Field(default=None, ge=0)
    total_cost: float | None = Field(default=None, ge=0)


class StandardizedOutput(BaseModel):
    """Canonical output.jsonl schema for all benchmarks."""

    model_config = ConfigDict(extra="forbid")

    instance_id: str
    attempt: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    status: Literal["success", "error", "skipped"]
    resolved: bool | None = None
    error: str | None = None
    test_result: dict[str, Any] = Field(default_factory=dict)
    cost: CostBreakdown = Field(default_factory=CostBreakdown)
    artifacts_url: str


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


def write_output_line(output_path: Path, output: StandardizedOutput) -> None:
    """Append one standardized line to a JSONL file with a file lock."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(output.model_dump_json() + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def load_output_file(path: str | Path) -> list[StandardizedOutput]:
    """Load a canonical output.jsonl into typed objects."""
    outputs: list[StandardizedOutput] = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Output file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                outputs.append(StandardizedOutput.model_validate_json(line))
            except ValidationError as exc:
                raise ValidationError(
                    f"Invalid line {line_num} in {path}: {exc}"
                ) from exc
    return outputs


def validate_output_file(path: str | Path) -> None:
    """Validate that every line in the file conforms to the schema."""
    load_output_file(path)


def select_best_attempts(
    outputs: Sequence[StandardizedOutput],
) -> dict[str, StandardizedOutput]:
    """
    Collapse attempts per instance, preferring successful attempts.

    Picks the highest-attempt successful line if present,
    otherwise the highest-attempt line.
    """
    best: dict[str, StandardizedOutput] = {}
    for out in sorted(outputs, key=lambda o: (o.instance_id, o.attempt)):
        existing = best.get(out.instance_id)
        if existing is None:
            best[out.instance_id] = out
            continue

        if out.resolved:
            if not existing.resolved or out.attempt >= existing.attempt:
                best[out.instance_id] = out
            continue

        if out.attempt >= existing.attempt and not existing.resolved:
            best[out.instance_id] = out
    return best


def derive_report(outputs: Iterable[StandardizedOutput]) -> dict[str, Any]:
    """Generate a derived summary report from canonical outputs."""
    outputs = list(outputs)
    best = select_best_attempts(outputs)
    resolved_ids = [k for k, v in best.items() if v.resolved]
    unresolved_ids = [k for k, v in best.items() if v.resolved is False]
    skipped_ids = [k for k, v in best.items() if v.status == "skipped"]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "derived_from": "output.jsonl",
        "totals": {
            "attempt_lines": len(outputs),
            "instances": len(best),
            "resolved": len(resolved_ids),
            "unresolved": len(unresolved_ids),
            "skipped": len(skipped_ids),
        },
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "skipped_ids": skipped_ids,
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
