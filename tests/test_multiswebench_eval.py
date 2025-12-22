"""Tests for Multi-SWE-Bench evaluation report file handling."""

from pathlib import Path

import pytest

from benchmarks.multiswebench.eval_infer import get_report_output_path


class TestGetReportOutputPath:
    """Tests for get_report_output_path function."""

    def test_jsonl_extension(self):
        """Test that .jsonl extension is replaced with .report.json."""
        input_path = "/path/to/output.jsonl"
        result = get_report_output_path(input_path)
        assert result == Path("/path/to/output.report.json")

    def test_json_extension(self):
        """Test that .json extension is replaced with .report.json."""
        input_path = "/path/to/output.json"
        result = get_report_output_path(input_path)
        assert result == Path("/path/to/output.report.json")

    def test_path_object_input(self):
        """Test that Path objects are handled correctly."""
        input_path = Path("/path/to/output.jsonl")
        result = get_report_output_path(input_path)
        assert result == Path("/path/to/output.report.json")

    def test_preserves_directory(self):
        """Test that the directory path is preserved."""
        input_path = "/some/deep/nested/directory/output.jsonl"
        result = get_report_output_path(input_path)
        assert result.parent == Path("/some/deep/nested/directory")
        assert result.name == "output.report.json"

    def test_relative_path(self):
        """Test that relative paths work correctly."""
        input_path = "output.jsonl"
        result = get_report_output_path(input_path)
        assert result == Path("output.report.json")

    def test_complex_filename(self):
        """Test filename with multiple dots."""
        input_path = "/path/to/model.v1.output.jsonl"
        result = get_report_output_path(input_path)
        assert result == Path("/path/to/model.v1.output.report.json")


@pytest.fixture
def temp_eval_structure(tmp_path):
    """Create a temporary directory structure mimicking eval output."""
    # Create input file
    input_file = tmp_path / "output.jsonl"
    input_file.write_text('{"test": "data"}\n')

    # Create eval_files/dataset directory structure
    eval_dir = tmp_path / "eval_files" / "dataset"
    eval_dir.mkdir(parents=True)

    # Create final_report.json
    report_file = eval_dir / "final_report.json"
    report_file.write_text('{"resolved": 5, "total": 10}')

    return {
        "input_file": input_file,
        "report_file": report_file,
        "expected_output": tmp_path / "output.report.json",
    }


def test_report_file_move_integration(temp_eval_structure, monkeypatch):
    """Test that the report file is moved to the correct location."""
    import shutil

    from benchmarks.multiswebench.eval_infer import get_report_output_path

    input_file = temp_eval_structure["input_file"]
    report_file = temp_eval_structure["report_file"]
    expected_output = temp_eval_structure["expected_output"]

    # Verify initial state
    assert report_file.exists()
    assert not expected_output.exists()

    # Simulate the move operation from main()
    output_report_path = get_report_output_path(str(input_file))
    if report_file.exists():
        shutil.move(str(report_file), str(output_report_path))

    # Verify final state
    assert not report_file.exists()
    assert expected_output.exists()
    assert expected_output.read_text() == '{"resolved": 5, "total": 10}'
