"""Tests for SWT-Bench evaluation script report file handling."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.swtbench.eval_infer import (
    convert_to_swtbench_format,
    run_swtbench_evaluation,
)


@pytest.fixture
def sample_openhands_output():
    """Create sample OpenHands output data."""
    return [
        {
            "instance_id": "django__django-11333",
            "test_result": {"git_patch": "diff --git a/file.py b/file.py\n..."},
            "instruction": "Fix the bug",
            "error": None,
            "history": [],
        },
        {
            "instance_id": "django__django-11334",
            "test_result": {"git_patch": "diff --git a/other.py b/other.py\n..."},
            "instruction": "Fix another bug",
            "error": None,
            "history": [],
        },
    ]


@pytest.fixture
def temp_input_file(sample_openhands_output):
    """Create a temporary input file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for entry in sample_openhands_output:
            f.write(json.dumps(entry) + "\n")
        return Path(f.name)


def test_convert_to_swtbench_format(temp_input_file):
    """Test conversion from OpenHands format to SWT-Bench format."""
    output_file = temp_input_file.with_suffix(".swtbench.jsonl")

    try:
        convert_to_swtbench_format(
            str(temp_input_file), str(output_file), model_name="test-model"
        )

        # Verify output file exists and has correct format
        assert output_file.exists()

        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["instance_id"] == "django__django-11333"
        assert entry1["model_name_or_path"] == "test-model"
        assert "model_patch" in entry1

        entry2 = json.loads(lines[1])
        assert entry2["instance_id"] == "django__django-11334"
        assert entry2["model_name_or_path"] == "test-model"

    finally:
        # Cleanup
        temp_input_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)


def test_run_swtbench_evaluation_returns_report_path():
    """Test that run_swtbench_evaluation returns the report file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a mock predictions file
        predictions_file = tmpdir_path / "output.swtbench.jsonl"
        predictions_file.write_text('{"instance_id": "test", "model_patch": ""}\n')

        model_name = "test-model"
        run_id = f"eval_{predictions_file.stem}"
        expected_report_name = f"{model_name}.{run_id}.json"

        # Create a mock swt-bench directory structure
        cache_dir = Path.home() / ".cache" / "openhands" / "swt-bench"
        swt_bench_dir = cache_dir / "swt-bench"
        swt_bench_dir.mkdir(parents=True, exist_ok=True)

        expected_report_path = swt_bench_dir / expected_report_name

        # Create the report file that swtbench harness would create
        expected_report_path.write_text('{"resolved": 0, "total": 1}')

        # Mock subprocess.run to simulate successful evaluation
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "/usr/bin/python"

        with patch("subprocess.run", return_value=mock_result):
            result = run_swtbench_evaluation(
                str(predictions_file),
                dataset="test-dataset",
                workers="1",
                model_name=model_name,
            )

        assert result is not None
        assert result == expected_report_path
        assert result.exists()

        # Cleanup
        expected_report_path.unlink(missing_ok=True)


def test_run_swtbench_evaluation_returns_none_when_report_not_found():
    """Test that run_swtbench_evaluation returns None when report file is not found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a mock predictions file
        predictions_file = tmpdir_path / "output.swtbench.jsonl"
        predictions_file.write_text('{"instance_id": "test", "model_patch": ""}\n')

        # Ensure the swt-bench directory exists but no report file
        cache_dir = Path.home() / ".cache" / "openhands" / "swt-bench"
        swt_bench_dir = cache_dir / "swt-bench"
        swt_bench_dir.mkdir(parents=True, exist_ok=True)

        # Mock subprocess.run to simulate successful evaluation
        # but don't create the report file
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "/usr/bin/python"

        with patch("subprocess.run", return_value=mock_result):
            result = run_swtbench_evaluation(
                str(predictions_file),
                dataset="test-dataset",
                workers="1",
                model_name="nonexistent-model",
            )

        assert result is None


def test_run_swtbench_evaluation_handles_model_name_with_slash():
    """Test that model names with slashes are handled correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a mock predictions file
        predictions_file = tmpdir_path / "output.swtbench.jsonl"
        predictions_file.write_text('{"instance_id": "test", "model_patch": ""}\n')

        model_name = "org/model-name"
        run_id = f"eval_{predictions_file.stem}"
        # Slashes should be replaced with double underscores
        expected_report_name = f"{model_name.replace('/', '__')}.{run_id}.json"

        # Create a mock swt-bench directory structure
        cache_dir = Path.home() / ".cache" / "openhands" / "swt-bench"
        swt_bench_dir = cache_dir / "swt-bench"
        swt_bench_dir.mkdir(parents=True, exist_ok=True)

        expected_report_path = swt_bench_dir / expected_report_name

        # Create the report file that swtbench harness would create
        expected_report_path.write_text('{"resolved": 0, "total": 1}')

        # Mock subprocess.run to simulate successful evaluation
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "/usr/bin/python"

        with patch("subprocess.run", return_value=mock_result):
            result = run_swtbench_evaluation(
                str(predictions_file),
                dataset="test-dataset",
                workers="1",
                model_name=model_name,
            )

        assert result is not None
        assert result.name == expected_report_name
        assert "__" in result.name  # Verify slash was replaced

        # Cleanup
        expected_report_path.unlink(missing_ok=True)
