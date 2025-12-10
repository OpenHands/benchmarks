"""Tests for incomplete evaluation run detection."""

import json
import tempfile
from pathlib import Path

from benchmarks.utils.incomplete_detection import IncompleteRunDetector


def test_get_completed_instances_empty_file():
    """Test getting completed instances from an empty output file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_dir = Path(tmpdir)
        output_file = eval_dir / "output.jsonl"
        output_file.touch()

        detector = IncompleteRunDetector(str(eval_dir))
        completed = detector.get_completed_instances("output.jsonl")

        assert completed == set()


def test_get_completed_instances_with_data():
    """Test getting completed instances from output file with data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_dir = Path(tmpdir)
        output_file = eval_dir / "output.jsonl"

        # Write some test data
        instances = [
            {"instance_id": "test-1", "result": "pass"},
            {"instance_id": "test-2", "result": "fail"},
            {"instance_id": "test-3", "result": "error"},
        ]

        with open(output_file, "w") as f:
            for inst in instances:
                f.write(json.dumps(inst) + "\n")

        detector = IncompleteRunDetector(str(eval_dir))
        completed = detector.get_completed_instances("output.jsonl")

        assert completed == {"test-1", "test-2", "test-3"}


def test_get_completed_instances_missing_file():
    """Test getting completed instances when output file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_dir = Path(tmpdir)

        detector = IncompleteRunDetector(str(eval_dir))
        completed = detector.get_completed_instances("output.jsonl")

        assert completed == set()


def test_get_completed_instances_invalid_json():
    """Test getting completed instances with invalid JSON lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_dir = Path(tmpdir)
        output_file = eval_dir / "output.jsonl"

        # Write some test data with invalid lines
        with open(output_file, "w") as f:
            f.write('{"instance_id": "test-1"}\n')
            f.write("invalid json line\n")
            f.write('{"instance_id": "test-2"}\n')

        detector = IncompleteRunDetector(str(eval_dir))
        completed = detector.get_completed_instances("output.jsonl")

        # Should skip invalid line and get the valid ones
        assert completed == {"test-1", "test-2"}


def test_check_completeness_complete():
    """Test completeness check when all instances are present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_dir = Path(tmpdir)
        output_file = eval_dir / "output.jsonl"

        # Write complete data
        expected = {"test-1", "test-2", "test-3"}
        with open(output_file, "w") as f:
            for inst_id in expected:
                f.write(json.dumps({"instance_id": inst_id}) + "\n")

        detector = IncompleteRunDetector(str(eval_dir))
        result = detector.check_completeness(expected, "output.jsonl")

        assert result["is_complete"] is True
        assert result["expected_count"] == 3
        assert result["completed_count"] == 3
        assert result["missing_count"] == 0
        assert result["missing_instances"] == []


def test_check_completeness_incomplete():
    """Test completeness check when some instances are missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_dir = Path(tmpdir)
        output_file = eval_dir / "output.jsonl"

        # Write incomplete data (missing test-3)
        expected = {"test-1", "test-2", "test-3"}
        completed = {"test-1", "test-2"}

        with open(output_file, "w") as f:
            for inst_id in completed:
                f.write(json.dumps({"instance_id": inst_id}) + "\n")

        detector = IncompleteRunDetector(str(eval_dir))
        result = detector.check_completeness(expected, "output.jsonl")

        assert result["is_complete"] is False
        assert result["expected_count"] == 3
        assert result["completed_count"] == 2
        assert result["missing_count"] == 1
        assert result["missing_instances"] == ["test-3"]


def test_check_completeness_with_extra():
    """Test completeness check with extra unexpected instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_dir = Path(tmpdir)
        output_file = eval_dir / "output.jsonl"

        # Write data with extra instance
        expected = {"test-1", "test-2"}
        completed = {"test-1", "test-2", "test-extra"}

        with open(output_file, "w") as f:
            for inst_id in completed:
                f.write(json.dumps({"instance_id": inst_id}) + "\n")

        detector = IncompleteRunDetector(str(eval_dir))
        result = detector.check_completeness(expected, "output.jsonl")

        assert result["is_complete"] is True  # All expected are present
        assert result["expected_count"] == 2
        assert result["completed_count"] == 3
        assert result["extra_count"] == 1
        assert result["extra_instances"] == ["test-extra"]


def test_report_incomplete_run_complete(caplog):
    """Test reporting for a complete run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_dir = Path(tmpdir)
        output_file = eval_dir / "output.jsonl"

        expected = {"test-1", "test-2"}
        with open(output_file, "w") as f:
            for inst_id in expected:
                f.write(json.dumps({"instance_id": inst_id}) + "\n")

        detector = IncompleteRunDetector(str(eval_dir))
        detector.report_incomplete_run(expected, "output.jsonl")

        # Should log that run is complete
        assert "complete" in caplog.text.lower()


def test_report_incomplete_run_incomplete(caplog):
    """Test reporting for an incomplete run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_dir = Path(tmpdir)
        output_file = eval_dir / "output.jsonl"

        expected = {"test-1", "test-2", "test-3"}
        completed = {"test-1"}

        with open(output_file, "w") as f:
            for inst_id in completed:
                f.write(json.dumps({"instance_id": inst_id}) + "\n")

        detector = IncompleteRunDetector(str(eval_dir))
        detector.report_incomplete_run(expected, "output.jsonl")

        # Should log error about incomplete run
        assert "INCOMPLETE" in caplog.text
        assert "test-2" in caplog.text
        assert "test-3" in caplog.text
