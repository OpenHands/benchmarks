"""Unit tests for TraceTrial-Bench report aggregation (eval_infer)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.tracetrialbench.eval_infer import (
    main as eval_main,
    process_tracetrialbench_results,
)


def _write_sample_output(path: Path) -> None:
    rows = [
        {
            "instance_id": "C1-escalation-export-001",
            "error": None,
            "test_result": {
                "passed": True,
                "agent_pass": True,
                "harness_pass": True,
                "clean_pass": True,
                "agent_score_pct": 100.0,
                "harness_score_pct": 100.0,
                "overall_score": 100.0,
                "attribution": "clean_pass",
                "dimension": "C",
            },
            "metrics": {
                "accumulated_cost": 0.02,
                "total_prompt_tokens": 1000,
                "total_completion_tokens": 200,
            },
        },
        {
            "instance_id": "T3-reserved-tool-name-001",
            "error": None,
            "test_result": {
                "passed": True,
                "agent_pass": True,
                "harness_pass": False,
                "clean_pass": False,
                "agent_score_pct": 100.0,
                "harness_score_pct": 50.0,
                "overall_score": 75.0,
                "attribution": "harness_defect_absorbed",
                "dimension": "T",
            },
            "metrics": {
                "accumulated_cost": 0.03,
                "total_prompt_tokens": 1500,
                "total_completion_tokens": 300,
            },
        },
        {
            "instance_id": "E3-firmware-timeout-001",
            "error": "Verifier failed to start",
            "test_result": {},
            "metrics": {
                "accumulated_cost": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
            },
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_process_tracetrialbench_results_agent_pass_vs_clean_pass(tmp_path: Path):
    output_jsonl = tmp_path / "output.jsonl"
    _write_sample_output(output_jsonl)

    agent_report_path = tmp_path / "agent.report.json"
    agent_report = process_tracetrialbench_results(
        str(output_jsonl),
        str(agent_report_path),
        criterion="agent_pass",
    )
    assert agent_report["total_instances"] == 3
    assert agent_report["completed_instances"] == 2
    assert agent_report["incomplete_instances"] == 1
    assert agent_report["error_instances"] == 1
    assert agent_report["resolved_instances"] == 2
    assert agent_report["resolved_ids"] == [
        "C1-escalation-export-001",
        "T3-reserved-tool-name-001",
    ]
    assert agent_report["aggregate_metrics"]["total_cost_usd"] == pytest.approx(0.05)
    assert agent_report["aggregate_metrics"]["total_prompt_tokens"] == 2500
    assert agent_report["aggregate_metrics"]["total_completion_tokens"] == 500

    tt = agent_report["tracetrial_metrics"]
    assert tt["agent_pass_rate"] == 100.0
    assert tt["clean_pass_rate"] == 50.0
    assert tt["agent_score_pct"] == 100.0
    assert tt["harness_macro_pct"] == 75.0
    assert tt["overall_score_pct"] == 87.5
    assert tt["per_dimension"]["C"]["harness_score_pct"] == 100.0
    assert tt["per_dimension"]["T"]["harness_score_pct"] == 50.0
    assert tt["attribution_counts"] == {
        "clean_pass": 1,
        "harness_defect_absorbed": 1,
    }

    clean_report_path = tmp_path / "clean.report.json"
    clean_report = process_tracetrialbench_results(
        str(output_jsonl),
        str(clean_report_path),
        criterion="clean_pass",
    )
    assert clean_report["resolved_instances"] == 1
    assert clean_report["resolved_ids"] == ["C1-escalation-export-001"]
    assert clean_report["unresolved_ids"] == ["T3-reserved-tool-name-001"]


def test_eval_main_writes_default_report_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    output_jsonl = tmp_path / "output.jsonl"
    _write_sample_output(output_jsonl)

    eval_main([str(output_jsonl)])
    expected_report = output_jsonl.with_suffix(".report.json")
    assert expected_report.is_file()
    stdout_lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    assert json.loads(stdout_lines[-1]) == {"report_json": str(expected_report)}
