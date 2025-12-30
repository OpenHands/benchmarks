from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# Metric references:
# - SWE-bench report format: https://github.com/princeton-nlp/SWE-bench/blob/main/swebench/harness/reporting.py
# - SWT-bench resolution logic: https://github.com/logic-star-ai/swt-bench/blob/master/src/grading.py
# - SWT-bench report aggregates: https://github.com/logic-star-ai/swt-bench/blob/master/src/run_evaluation.py
# - Commit0 evaluation (average pass rate): https://github.com/commit-0/commit0/blob/main/commit0/harness/evaluate.py
# - GAIA accuracy definition: https://arxiv.org/abs/2311.12983
# - GAIA dataset: https://huggingface.co/datasets/gaia-benchmark/GAIA

ReportT = TypeVar("ReportT", bound=BaseModel)


class SwebenchReport(BaseModel):
    """SWE-bench compatible evaluation summary."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model_name_or_path: str | None = None

    total_instances: int = Field(
        ge=0, description="Total number of instances in the benchmark split."
    )
    submitted_instances: int = Field(
        ge=0, description="Number of instances submitted for evaluation."
    )
    completed_instances: int = Field(
        ge=0, description="Number of instances with completed, non-error outputs."
    )
    resolved_instances: int = Field(
        ge=0, description="Number of instances marked as resolved/successful."
    )
    unresolved_instances: int = Field(
        ge=0, description="Number of instances marked as unresolved/failed."
    )
    empty_patch_instances: int = Field(
        ge=0, description="Number of instances producing an empty patch."
    )
    error_instances: int = Field(
        ge=0, description="Number of instances that failed with errors."
    )
    incomplete_instances: int | None = Field(
        default=None,
        ge=0,
        description="Number of instances that did not complete (optional).",
    )

    completed_ids: list[str] = Field(
        default_factory=list, description="Instance IDs with completed outputs."
    )
    incomplete_ids: list[str] = Field(
        default_factory=list,
        description="Instance IDs that did not complete or are missing outputs.",
    )
    submitted_ids: list[str] = Field(
        default_factory=list, description="Instance IDs submitted for evaluation."
    )
    resolved_ids: list[str] = Field(
        default_factory=list, description="Instance IDs marked as resolved."
    )
    unresolved_ids: list[str] = Field(
        default_factory=list, description="Instance IDs marked as unresolved."
    )
    empty_patch_ids: list[str] = Field(
        default_factory=list, description="Instance IDs with empty patches."
    )
    error_ids: list[str] = Field(
        default_factory=list, description="Instance IDs that failed with errors."
    )

    schema_version: int | None = 2
    unstopped_instances: int | None = None
    unstopped_containers: list[str] = Field(default_factory=list)
    unremoved_images: list[str] = Field(default_factory=list)

    @classmethod
    def from_ids(
        cls,
        *,
        total_instances: int,
        completed_ids: Sequence[str],
        resolved_ids: Sequence[str],
        unresolved_ids: Sequence[str],
        empty_patch_ids: Sequence[str] | None = None,
        error_ids: Sequence[str] | None = None,
        submitted_ids: Sequence[str] | None = None,
        incomplete_ids: Sequence[str] | None = None,
        model_name_or_path: str | None = None,
    ) -> "SwebenchReport":
        empty_patch_ids_list = list(empty_patch_ids or [])
        error_ids_list = list(error_ids or [])
        completed_ids_list = list(completed_ids)
        resolved_ids_list = list(resolved_ids)
        unresolved_ids_list = list(unresolved_ids)
        submitted_ids_list = (
            list(submitted_ids)
            if submitted_ids is not None
            else list(completed_ids_list)
        )
        incomplete_ids_list = list(incomplete_ids or [])

        return cls(
            model_name_or_path=model_name_or_path,
            total_instances=total_instances,
            submitted_instances=len(submitted_ids_list),
            completed_instances=len(completed_ids_list),
            resolved_instances=len(resolved_ids_list),
            unresolved_instances=len(unresolved_ids_list),
            empty_patch_instances=len(empty_patch_ids_list),
            error_instances=len(error_ids_list),
            incomplete_instances=(
                len(incomplete_ids_list) if incomplete_ids_list else None
            ),
            completed_ids=completed_ids_list,
            incomplete_ids=incomplete_ids_list,
            submitted_ids=submitted_ids_list,
            resolved_ids=resolved_ids_list,
            unresolved_ids=unresolved_ids_list,
            empty_patch_ids=empty_patch_ids_list,
            error_ids=error_ids_list,
        )

    @classmethod
    def from_swtbench_report(cls, report: Mapping[str, Any]) -> "SwebenchReport":
        completed_ids = list(report.get("completed_ids", []))
        resolved_ids = list(report.get("resolved_ids", []))
        unresolved_ids = list(report.get("unresolved_ids", []))
        error_ids = list(report.get("error_ids", []))
        total_instances = int(
            report.get(
                "total_instances",
                len(completed_ids) + len(unresolved_ids) + len(error_ids),
            )
        )

        return cls.from_ids(
            total_instances=total_instances,
            completed_ids=completed_ids,
            resolved_ids=resolved_ids,
            unresolved_ids=unresolved_ids,
            error_ids=error_ids,
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.write_text(
            self.model_dump_json(indent=4, by_alias=True, exclude_none=True)
        )


class Commit0InstanceMetrics(BaseModel):
    num_tests: int
    num_passed: int
    pass_rate: float


class Commit0Report(SwebenchReport):
    total_tests: int
    total_passed_tests: int
    instance_metrics: dict[str, Commit0InstanceMetrics] = Field(default_factory=dict)
    average_pass_rate: float | None = None


class GaiaReport(SwebenchReport):
    eval_limit: int | None = None


class SwtbenchReport(SwebenchReport):
    mean_coverage: float | None = Field(default=None, alias="Mean coverage")
    mean_coverage_delta: float | None = Field(default=None, alias="Mean coverage delta")


def write_report(path: Path, report: BaseModel, *, by_alias: bool = True) -> None:
    path.write_text(
        report.model_dump_json(indent=4, by_alias=by_alias, exclude_none=True)
    )


def read_report(path: Path, model: type[ReportT]) -> ReportT:
    return model.model_validate_json(path.read_text())


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
