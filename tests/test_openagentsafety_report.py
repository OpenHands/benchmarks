"""Tests for openagentsafety generate_report functionality."""

import json
import tempfile
from pathlib import Path

from benchmarks.openagentsafety.run_infer import generate_report


def test_mixed_results():
    """Resolved, unresolved, and error instances are classified correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_jsonl = Path(tmpdir) / "output.jsonl"
        report_path = Path(tmpdir) / "output.report.json"

        lines = [
            # resolved: result == total > 0
            {
                "instance_id": "resolved-1",
                "test_result": {"final_score": {"result": 3, "total": 3}},
            },
            # unresolved: result < total
            {
                "instance_id": "unresolved-1",
                "test_result": {"final_score": {"result": 1, "total": 3}},
            },
            # unresolved: result == 0
            {
                "instance_id": "unresolved-2",
                "test_result": {"final_score": {"result": 0, "total": 2}},
            },
            # error at top level
            {
                "instance_id": "error-1",
                "error": "timeout",
                "test_result": {},
            },
            # error inside test_result
            {
                "instance_id": "error-2",
                "test_result": {"error": "evaluation crashed"},
            },
        ]
        output_jsonl.write_text(
            "\n".join(json.dumps(l) for l in lines) + "\n"
        )

        generate_report(str(output_jsonl), str(report_path), "test-model")

        report = json.loads(report_path.read_text())

        assert report["model_name_or_path"] == "test-model"
        assert report["resolved_instances"] == 1
        assert report["resolved_ids"] == ["resolved-1"]
        assert report["unresolved_instances"] == 2
        assert sorted(report["unresolved_ids"]) == ["unresolved-1", "unresolved-2"]
        assert report["error_instances"] == 2
        assert report["completed_instances"] == 3
        assert report["submitted_instances"] == 5
        assert report["total_instances"] == 5


def test_empty_file():
    """An empty output.jsonl produces a report with all zeroes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_jsonl = Path(tmpdir) / "output.jsonl"
        report_path = Path(tmpdir) / "output.report.json"
        output_jsonl.write_text("")

        generate_report(str(output_jsonl), str(report_path), "test-model")

        report = json.loads(report_path.read_text())
        assert report["total_instances"] == 0
        assert report["resolved_instances"] == 0
        assert report["unresolved_instances"] == 0
        assert report["error_instances"] == 0


def test_missing_file():
    """A missing output.jsonl produces no report file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "output.report.json"
        generate_report("/nonexistent/output.jsonl", str(report_path), "m")
        assert not report_path.exists()


def test_malformed_json_lines_skipped():
    """Malformed JSON lines are silently skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_jsonl = Path(tmpdir) / "output.jsonl"
        report_path = Path(tmpdir) / "output.report.json"

        content = "not valid json\n" + json.dumps(
            {
                "instance_id": "good-1",
                "test_result": {"final_score": {"result": 1, "total": 1}},
            }
        ) + "\n"
        output_jsonl.write_text(content)

        generate_report(str(output_jsonl), str(report_path), "test-model")

        report = json.loads(report_path.read_text())
        assert report["resolved_instances"] == 1
        assert report["resolved_ids"] == ["good-1"]
        assert report["total_instances"] == 1


def test_missing_final_score_is_unresolved():
    """An instance with no final_score is completed but unresolved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_jsonl = Path(tmpdir) / "output.jsonl"
        report_path = Path(tmpdir) / "output.report.json"

        output_jsonl.write_text(
            json.dumps({"instance_id": "no-score", "test_result": {}}) + "\n"
        )

        generate_report(str(output_jsonl), str(report_path), "test-model")

        report = json.loads(report_path.read_text())
        assert report["completed_instances"] == 1
        assert report["resolved_instances"] == 0
        assert report["unresolved_instances"] == 1
        assert report["unresolved_ids"] == ["no-score"]
