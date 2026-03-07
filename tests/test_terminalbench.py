"""Tests for Terminal-Bench benchmark module."""

import json
from pathlib import Path

import pytest

from benchmarks.terminalbench.eval_infer import process_terminalbench_results
from benchmarks.terminalbench.run_infer import convert_harbor_to_eval_output


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


class TestConvertHarborToEvalOutput:
    """Tests for convert_harbor_to_eval_output function."""

    def _create_harbor_structure(
        self, tmp_path: Path, trials: list[tuple[str, dict]]
    ) -> Path:
        """Create a mock Harbor output structure.

        Harbor stores results as:
            harbor_output/TIMESTAMP/TRIAL_NAME/result.json
        with a job-level result.json at harbor_output/TIMESTAMP/result.json
        """
        harbor_dir = tmp_path / "harbor_output"
        job_dir = harbor_dir / "2026-01-01__00-00-00"
        job_dir.mkdir(parents=True)

        # Create job-level result.json
        (job_dir / "result.json").write_text(json.dumps({"id": "test-job"}))

        for trial_name, trial_result in trials:
            trial_dir = job_dir / trial_name
            trial_dir.mkdir()
            (trial_dir / "result.json").write_text(json.dumps(trial_result))

        return harbor_dir

    def test_successful_trial_parsing(self, tmp_path: Path) -> None:
        """Test successful parsing of harbor trial result."""
        trial_result = {
            "task_name": "hello-world",
            "trial_name": "hello-world__abc123",
            "trial_uri": "file:///path/to/trial",
            "agent_result": {
                "n_input_tokens": 500,
                "n_output_tokens": 100,
                "cost_usd": 0.01,
            },
            "verifier_result": {"rewards": {"reward": 1.0}},
            "exception_info": None,
        }

        harbor_dir = self._create_harbor_structure(
            tmp_path, [("hello-world__abc123", trial_result)]
        )
        output_file = tmp_path / "output.jsonl"

        convert_harbor_to_eval_output(harbor_dir, output_file)

        assert output_file.exists()
        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 1
        assert entries[0]["instance_id"] == "hello-world"
        assert entries[0]["metrics"]["total_cost_usd"] == 0.01
        assert entries[0]["test_result"]["passed"] is True

    def test_failed_trial(self, tmp_path: Path) -> None:
        """Test parsing of a trial with reward 0."""
        trial_result = {
            "task_name": "test-task",
            "trial_name": "test-task__xyz",
            "agent_result": {
                "n_input_tokens": None,
                "n_output_tokens": None,
                "cost_usd": None,
            },
            "verifier_result": {"rewards": {"reward": 0.0}},
            "exception_info": None,
        }

        harbor_dir = self._create_harbor_structure(
            tmp_path, [("test-task__xyz", trial_result)]
        )
        output_file = tmp_path / "output.jsonl"

        convert_harbor_to_eval_output(harbor_dir, output_file)

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 1
        assert entries[0]["test_result"]["passed"] is False
        assert entries[0]["metrics"]["total_cost_usd"] == 0.0

    def test_trial_with_exception(self, tmp_path: Path) -> None:
        """Test handling of a trial with exception."""
        trial_result = {
            "task_name": "error-task",
            "trial_name": "error-task__err",
            "agent_result": {},
            "verifier_result": {},
            "exception_info": {"type": "TimeoutError", "message": "Agent timed out"},
        }

        harbor_dir = self._create_harbor_structure(
            tmp_path, [("error-task__err", trial_result)]
        )
        output_file = tmp_path / "output.jsonl"

        # Should raise since all trials have exceptions and none succeeded
        with pytest.raises(RuntimeError, match="All .* trials failed"):
            convert_harbor_to_eval_output(harbor_dir, output_file)

    def test_mixed_valid_and_exception_trials(self, tmp_path: Path) -> None:
        """Test handling mix of successful and exception trials."""
        trials = [
            (
                "good-task__abc",
                {
                    "task_name": "good-task",
                    "trial_name": "good-task__abc",
                    "agent_result": {},
                    "verifier_result": {"rewards": {"reward": 1.0}},
                    "exception_info": None,
                },
            ),
            (
                "bad-task__def",
                {
                    "task_name": "bad-task",
                    "trial_name": "bad-task__def",
                    "agent_result": {},
                    "verifier_result": {},
                    "exception_info": {"type": "Error", "message": "Failed"},
                },
            ),
        ]

        harbor_dir = self._create_harbor_structure(tmp_path, trials)
        output_file = tmp_path / "output.jsonl"
        convert_harbor_to_eval_output(harbor_dir, output_file)

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 2
        success = [e for e in entries if e.get("error") is None]
        errors = [e for e in entries if e.get("error") is not None]
        assert len(success) == 1
        assert len(errors) == 1

    def test_empty_job_directory(self, tmp_path: Path) -> None:
        """Test handling of empty harbor job directory."""
        harbor_dir = tmp_path / "harbor_output"
        job_dir = harbor_dir / "2026-01-01__00-00-00"
        job_dir.mkdir(parents=True)
        (job_dir / "result.json").write_text(json.dumps({"id": "test"}))

        output_file = tmp_path / "output.jsonl"

        with pytest.raises(RuntimeError, match="No trial result files found"):
            convert_harbor_to_eval_output(harbor_dir, output_file)

    def test_missing_job_directory(self, tmp_path: Path) -> None:
        """Test handling when no job directory exists."""
        harbor_dir = tmp_path / "harbor_output"
        harbor_dir.mkdir()

        output_file = tmp_path / "output.jsonl"

        with pytest.raises(RuntimeError, match="No harbor job directory found"):
            convert_harbor_to_eval_output(harbor_dir, output_file)

    def test_discovery_finds_all_trials(self, tmp_path: Path) -> None:
        """Test that discovery finds all trial subdirectories."""
        trials = [
            (
                f"task-{i}__trial{i}",
                {
                    "task_name": f"task-{i}",
                    "trial_name": f"task-{i}__trial{i}",
                    "agent_result": {},
                    "verifier_result": {"rewards": {"reward": 0.0}},
                    "exception_info": None,
                },
            )
            for i in range(5)
        ]

        harbor_dir = self._create_harbor_structure(tmp_path, trials)
        output_file = tmp_path / "output.jsonl"

        convert_harbor_to_eval_output(harbor_dir, output_file)

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 5
        instance_ids = {e["instance_id"] for e in entries}
        assert instance_ids == {f"task-{i}" for i in range(5)}
