import json
from pathlib import Path

from benchmarks.utils.models import (
    CostBreakdown,
    EvalOutput,
    load_output_file,
    select_best_attempts,
    write_derived_report,
    write_output_line,
)


def test_standardized_output_roundtrip(tmp_path: Path) -> None:
    output_path = tmp_path / "output.jsonl"
    original = EvalOutput(
        instance_id="abc",
        test_result={"git_patch": "diff --git"},
        attempt=1,
        max_attempts=3,
        status="success",
        resolved=True,
        duration_seconds=12.5,
        cost=CostBreakdown(total_cost=1.5, input_tokens=10, output_tokens=5),
        artifacts_url="artifacts/abc/attempt_1",
    )

    write_output_line(output_path, original)
    loaded = load_output_file(output_path)

    assert len(loaded) == 1
    assert loaded[0].instance_id == original.instance_id
    assert loaded[0].cost.total_cost == 1.5
    assert loaded[0].duration_seconds == 12.5


def test_select_best_attempts_prefers_success(tmp_path: Path) -> None:
    out1 = EvalOutput(
        instance_id="item",
        attempt=1,
        max_attempts=2,
        status="error",
        resolved=False,
        test_result={},
        cost=CostBreakdown(),
        artifacts_url="a",
    )
    out2 = EvalOutput(
        instance_id="item",
        attempt=2,
        max_attempts=2,
        status="success",
        resolved=True,
        test_result={},
        cost=CostBreakdown(),
        artifacts_url="b",
    )
    best = select_best_attempts([out1, out2])
    assert best["item"].attempt == 2
    assert best["item"].resolved


def test_write_derived_report(tmp_path: Path) -> None:
    out_file = tmp_path / "output.jsonl"
    write_output_line(
        out_file,
        EvalOutput(
            instance_id="one",
            attempt=1,
            max_attempts=1,
            status="success",
            resolved=True,
            test_result={},
            cost=CostBreakdown(),
            artifacts_url="artifacts/one/attempt_1",
        ),
    )
    write_output_line(
        out_file,
        EvalOutput(
            instance_id="two",
            attempt=1,
            max_attempts=1,
            status="error",
            resolved=False,
            test_result={},
            cost=CostBreakdown(),
            artifacts_url="artifacts/two/attempt_1",
        ),
    )

    report_path = write_derived_report(tmp_path)
    data = json.loads(report_path.read_text())
    assert data["totals"]["instances"] == 2
    assert data["totals"]["resolved"] == 1
    assert "resolved_ids" in data
