"""Tests for SkillsBench eval_infer module."""

import json
from pathlib import Path

from benchmarks.skillsbench.eval_infer import process_skillsbench_results


class TestProcessSkillsbenchResults:
    """Tests for the process_skillsbench_results function."""

    def test_empty_input(self, tmp_path: Path) -> None:
        """Test processing empty input file."""
        input_file = tmp_path / "empty.jsonl"
        output_file = tmp_path / "empty.report.json"
        input_file.write_text("")

        result = process_skillsbench_results(str(input_file), str(output_file))

        assert result["total_instances"] == 0
        assert result["completed_instances"] == 0
        assert result["resolved_instances"] == 0

    def test_resolved_instance(self, tmp_path: Path) -> None:
        """Test processing a resolved (passed=True) instance."""
        input_file = tmp_path / "resolved.jsonl"
        output_file = tmp_path / "resolved.report.json"

        entry = {
            "instance_id": "benchflow/weighted-gdp-calc",
            "test_result": {"passed": True, "rewards": {"reward": 1.0}},
            "error": None,
        }
        input_file.write_text(json.dumps(entry) + "\n")

        result = process_skillsbench_results(str(input_file), str(output_file))

        assert result["resolved_instances"] == 1
        assert result["unresolved_instances"] == 0
        assert "benchflow/weighted-gdp-calc" in result["resolved_ids"]

    def test_unresolved_instance(self, tmp_path: Path) -> None:
        """Test processing an unresolved (passed=False) instance."""
        input_file = tmp_path / "unresolved.jsonl"
        output_file = tmp_path / "unresolved.report.json"

        entry = {
            "instance_id": "benchflow/task-1",
            "test_result": {"passed": False, "rewards": {"reward": 0.0}},
            "error": None,
        }
        input_file.write_text(json.dumps(entry) + "\n")

        result = process_skillsbench_results(str(input_file), str(output_file))

        assert result["resolved_instances"] == 0
        assert result["unresolved_instances"] == 1

    def test_instance_with_error(self, tmp_path: Path) -> None:
        """Test processing an instance that errored."""
        input_file = tmp_path / "error.jsonl"
        output_file = tmp_path / "error.report.json"

        entry = {
            "instance_id": "benchflow/error-task",
            "test_result": {},
            "error": "ValueError: LLM_API_KEY environment variable must be set",
        }
        input_file.write_text(json.dumps(entry) + "\n")

        result = process_skillsbench_results(str(input_file), str(output_file))

        assert result["error_instances"] == 1
        assert result["incomplete_instances"] == 1
        assert result["completed_instances"] == 0
        assert "benchflow/error-task" in result["error_ids"]

    def test_multiple_instances(self, tmp_path: Path) -> None:
        """Test processing multiple instances with mixed results."""
        input_file = tmp_path / "multi.jsonl"
        output_file = tmp_path / "multi.report.json"

        entries = [
            {
                "instance_id": "benchflow/task-1",
                "test_result": {"passed": True},
                "error": None,
            },
            {
                "instance_id": "benchflow/task-2",
                "test_result": {"passed": False},
                "error": None,
            },
            {"instance_id": "benchflow/task-3", "test_result": {}, "error": "Timeout"},
        ]
        input_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = process_skillsbench_results(str(input_file), str(output_file))

        assert result["total_instances"] == 3
        assert result["completed_instances"] == 2
        assert result["resolved_instances"] == 1
        assert result["unresolved_instances"] == 1
        assert result["error_instances"] == 1

    def test_report_file_written(self, tmp_path: Path) -> None:
        """Test that report file is written correctly."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.report.json"

        entry = {
            "instance_id": "benchflow/task-1",
            "test_result": {"passed": True},
            "error": None,
        }
        input_file.write_text(json.dumps(entry) + "\n")

        process_skillsbench_results(str(input_file), str(output_file))

        assert output_file.exists()
        with open(output_file) as f:
            report = json.load(f)
        assert "total_instances" in report
        assert "resolved_ids" in report
        assert "aggregate_metrics" in report
