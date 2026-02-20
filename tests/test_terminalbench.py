"""Tests for Terminal-Bench benchmark module."""

import json
from pathlib import Path

from benchmarks.terminalbench.eval_infer import process_terminalbench_results


class TestProcessTerminalbenchResults:
    """Tests for the process_terminalbench_results function."""

    def test_empty_input(self, tmp_path: Path) -> None:
        """Test processing empty input file."""
        input_file = tmp_path / "empty.jsonl"
        output_file = tmp_path / "empty.report.json"
        input_file.write_text("")

        result = process_terminalbench_results(str(input_file), str(output_file))

        assert result["total_instances"] == 0
        assert result["completed_instances"] == 0
        assert result["resolved_instances"] == 0

    def test_single_completed_instance(self, tmp_path: Path) -> None:
        """Test processing a single completed instance."""
        input_file = tmp_path / "single.jsonl"
        output_file = tmp_path / "single.report.json"

        entry = {
            "instance_id": "hello-world",
            "test_result": {
                "trajectory_path": "/path/to/trajectory.json",
                "total_steps": 5,
                "final_metrics": {
                    "total_prompt_tokens": 1000,
                    "total_completion_tokens": 200,
                    "total_cost_usd": 0.01,
                },
            },
            "instruction": "Create hello.txt",
            "error": None,
            "history": [],
        }
        input_file.write_text(json.dumps(entry) + "\n")

        result = process_terminalbench_results(str(input_file), str(output_file))

        assert result["total_instances"] == 1
        assert result["completed_instances"] == 1
        # Without explicit passed=True, instance is unresolved
        assert result["unresolved_instances"] == 1
        assert result["resolved_instances"] == 0
        assert "hello-world" in result["completed_ids"]

    def test_resolved_instance(self, tmp_path: Path) -> None:
        """Test processing a resolved (passed=True) instance."""
        input_file = tmp_path / "resolved.jsonl"
        output_file = tmp_path / "resolved.report.json"

        entry = {
            "instance_id": "test-task",
            "test_result": {
                "passed": True,
                "total_steps": 10,
            },
            "instruction": "Do something",
            "error": None,
            "history": [],
        }
        input_file.write_text(json.dumps(entry) + "\n")

        result = process_terminalbench_results(str(input_file), str(output_file))

        assert result["resolved_instances"] == 1
        assert result["unresolved_instances"] == 0
        assert "test-task" in result["resolved_ids"]

    def test_instance_with_error(self, tmp_path: Path) -> None:
        """Test processing an instance with error."""
        input_file = tmp_path / "error.jsonl"
        output_file = tmp_path / "error.report.json"

        entry = {
            "instance_id": "error-task",
            "test_result": {},
            "instruction": "Do something",
            "error": "Runtime timeout",
            "history": [],
        }
        input_file.write_text(json.dumps(entry) + "\n")

        result = process_terminalbench_results(str(input_file), str(output_file))

        assert result["error_instances"] == 1
        assert result["incomplete_instances"] == 1
        assert result["completed_instances"] == 0
        assert "error-task" in result["error_ids"]

    def test_multiple_instances(self, tmp_path: Path) -> None:
        """Test processing multiple instances."""
        input_file = tmp_path / "multi.jsonl"
        output_file = tmp_path / "multi.report.json"

        entries = [
            {
                "instance_id": "task-1",
                "test_result": {"passed": True},
                "error": None,
            },
            {
                "instance_id": "task-2",
                "test_result": {"passed": False},
                "error": None,
            },
            {
                "instance_id": "task-3",
                "test_result": {},
                "error": "Failed",
            },
        ]
        input_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = process_terminalbench_results(str(input_file), str(output_file))

        assert result["total_instances"] == 3
        assert result["completed_instances"] == 2
        assert result["resolved_instances"] == 1
        assert result["unresolved_instances"] == 1
        assert result["error_instances"] == 1

    def test_aggregate_metrics(self, tmp_path: Path) -> None:
        """Test that metrics are aggregated correctly."""
        input_file = tmp_path / "metrics.jsonl"
        output_file = tmp_path / "metrics.report.json"

        entries = [
            {
                "instance_id": "task-1",
                "test_result": {
                    "final_metrics": {
                        "total_prompt_tokens": 1000,
                        "total_completion_tokens": 200,
                        "total_cost_usd": 0.01,
                    }
                },
                "metrics": {},
                "error": None,
            },
            {
                "instance_id": "task-2",
                "test_result": {},
                "metrics": {
                    "total_prompt_tokens": 2000,
                    "total_completion_tokens": 400,
                    "total_cost_usd": 0.02,
                },
                "error": None,
            },
        ]
        input_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = process_terminalbench_results(str(input_file), str(output_file))

        aggregate = result["aggregate_metrics"]
        assert aggregate["total_prompt_tokens"] == 3000
        assert aggregate["total_completion_tokens"] == 600
        assert abs(aggregate["total_cost_usd"] - 0.03) < 0.001

    def test_duplicate_instance_ids_ignored(self, tmp_path: Path) -> None:
        """Test that duplicate instance IDs are handled."""
        input_file = tmp_path / "dup.jsonl"
        output_file = tmp_path / "dup.report.json"

        entries = [
            {"instance_id": "task-1", "test_result": {}, "error": None},
            {"instance_id": "task-1", "test_result": {}, "error": None},  # Duplicate
        ]
        input_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = process_terminalbench_results(str(input_file), str(output_file))

        # Only first occurrence should be counted
        assert result["completed_instances"] == 1

    def test_report_file_written(self, tmp_path: Path) -> None:
        """Test that report file is written correctly."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.report.json"

        entry = {
            "instance_id": "task-1",
            "test_result": {"passed": True},
            "error": None,
        }
        input_file.write_text(json.dumps(entry) + "\n")

        process_terminalbench_results(str(input_file), str(output_file))

        assert output_file.exists()
        with open(output_file) as f:
            report = json.load(f)
        assert "total_instances" in report
        assert "resolved_ids" in report


class TestConvertHarborOutput:
    """Tests for harbor output conversion functions."""

    def test_atif_trajectory_parsing(self, tmp_path: Path) -> None:
        """Test parsing of ATIF trajectory format."""
        # Create a mock ATIF trajectory
        trajectory = {
            "schema_version": "ATIF-v1.5",
            "session_id": "test-session-123",
            "agent": {"name": "openhands-sdk", "version": "1.0.0"},
            "steps": [
                {
                    "step_id": 1,
                    "source": "user",
                    "message": "Create a file",
                    "timestamp": "2025-01-01T00:00:00Z",
                },
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "I'll create the file",
                    "timestamp": "2025-01-01T00:00:01Z",
                },
            ],
            "final_metrics": {
                "total_prompt_tokens": 500,
                "total_completion_tokens": 100,
                "total_cost_usd": 0.005,
            },
        }

        traj_file = tmp_path / "trajectory.json"
        traj_file.write_text(json.dumps(trajectory))

        # Parse the trajectory
        with open(traj_file) as f:
            parsed = json.load(f)

        assert parsed["session_id"] == "test-session-123"
        assert len(parsed["steps"]) == 2
        assert parsed["final_metrics"]["total_cost_usd"] == 0.005
