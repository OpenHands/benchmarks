from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

ReportT = TypeVar("ReportT", bound=BaseModel)


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
