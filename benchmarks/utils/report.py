from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field


class SwebenchReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total_instances: int = Field(ge=0)
    submitted_instances: int = Field(ge=0)
    completed_instances: int = Field(ge=0)
    resolved_instances: int = Field(ge=0)
    unresolved_instances: int = Field(ge=0)
    empty_patch_instances: int = Field(ge=0)
    error_instances: int = Field(ge=0)

    completed_ids: list[str] = Field(default_factory=list)
    incomplete_ids: list[str] = Field(default_factory=list)
    submitted_ids: list[str] = Field(default_factory=list)
    resolved_ids: list[str] = Field(default_factory=list)
    unresolved_ids: list[str] = Field(default_factory=list)
    empty_patch_ids: list[str] = Field(default_factory=list)
    error_ids: list[str] = Field(default_factory=list)

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
            total_instances=total_instances,
            submitted_instances=len(submitted_ids_list),
            completed_instances=len(completed_ids_list),
            resolved_instances=len(resolved_ids_list),
            unresolved_instances=len(unresolved_ids_list),
            empty_patch_instances=len(empty_patch_ids_list),
            error_instances=len(error_ids_list),
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
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)


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
