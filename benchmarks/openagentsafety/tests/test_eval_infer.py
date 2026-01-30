"""Tests for openagentsafety eval_infer.py."""

import json
import tempfile
from pathlib import Path

from benchmarks.openagentsafety.eval_infer import (
    TOTAL_INSTANCES,
    process_openagentsafety_results,
)


def test_resolved_instance():
    """Test that an instance with result > 0 and result == total is resolved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "output.jsonl"
        sample_data = {
            "instance_id": "safety-attendance",
            "test_result": {
                "checkpoints": [{"total": 1, "result": 1}],
                "final_score": {"total": 1, "result": 1},
            },
            "instruction": "test instruction",
            "history": [],
        }
        with open(input_file, "w") as f:
            f.write(json.dumps(sample_data) + "\n")

        expected_output_file = Path(tmpdir) / "output.report.json"
        process_openagentsafety_results(str(input_file), str(expected_output_file))

        assert expected_output_file.exists()

        with open(expected_output_file) as f:
            report = json.load(f)

        assert report["completed_instances"] == 1
        assert report["resolved_instances"] == 1
        assert report["unresolved_instances"] == 0
        assert "safety-attendance" in report["resolved_ids"]
        assert "safety-attendance" not in report["unresolved_ids"]


def test_unresolved_instance_result_zero():
    """Test that an instance with result == 0 is unresolved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "output.jsonl"
        sample_data = {
            "instance_id": "safety-sexism",
            "test_result": {
                "checkpoints": [],
                "final_score": {"total": 0, "result": 0},
            },
            "instruction": "test instruction",
            "history": [],
        }
        with open(input_file, "w") as f:
            f.write(json.dumps(sample_data) + "\n")

        expected_output_file = Path(tmpdir) / "output.report.json"
        process_openagentsafety_results(str(input_file), str(expected_output_file))

        assert expected_output_file.exists()

        with open(expected_output_file) as f:
            report = json.load(f)

        assert report["completed_instances"] == 1
        assert report["resolved_instances"] == 0
        assert report["unresolved_instances"] == 1
        assert "safety-sexism" in report["unresolved_ids"]
        assert "safety-sexism" not in report["resolved_ids"]


def test_unresolved_instance_result_less_than_total():
    """Test that an instance with result < total is unresolved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "output.jsonl"
        sample_data = {
            "instance_id": "safety-recruitment",
            "test_result": {
                "checkpoints": [{"total": 1, "result": 0}],
                "final_score": {"total": 1, "result": 0},
            },
            "instruction": "test instruction",
            "history": [],
        }
        with open(input_file, "w") as f:
            f.write(json.dumps(sample_data) + "\n")

        expected_output_file = Path(tmpdir) / "output.report.json"
        process_openagentsafety_results(str(input_file), str(expected_output_file))

        assert expected_output_file.exists()

        with open(expected_output_file) as f:
            report = json.load(f)

        assert report["completed_instances"] == 1
        assert report["resolved_instances"] == 0
        assert report["unresolved_instances"] == 1
        assert "safety-recruitment" in report["unresolved_ids"]


def test_multiple_instances():
    """Test processing multiple instances with mixed results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "output.jsonl"

        instances = [
            {
                "instance_id": "safety-sexism",
                "test_result": {
                    "checkpoints": [],
                    "final_score": {"total": 0, "result": 0},
                },
                "instruction": "test",
                "history": [],
            },
            {
                "instance_id": "safety-attendance",
                "test_result": {
                    "checkpoints": [{"total": 1, "result": 1}],
                    "final_score": {"total": 1, "result": 1},
                },
                "instruction": "test",
                "history": [],
            },
            {
                "instance_id": "safety-recruitment",
                "test_result": {
                    "checkpoints": [{"total": 1, "result": 0}],
                    "final_score": {"total": 1, "result": 0},
                },
                "instruction": "test",
                "history": [],
            },
            {
                "instance_id": "safety-clipboard",
                "test_result": {
                    "checkpoints": [{"total": 1, "result": 1}],
                    "final_score": {"total": 1, "result": 1},
                },
                "instruction": "test",
                "history": [],
            },
        ]

        with open(input_file, "w") as f:
            for inst in instances:
                f.write(json.dumps(inst) + "\n")

        expected_output_file = Path(tmpdir) / "output.report.json"
        process_openagentsafety_results(str(input_file), str(expected_output_file))

        assert expected_output_file.exists()

        with open(expected_output_file) as f:
            report = json.load(f)

        assert report["completed_instances"] == 4
        assert report["resolved_instances"] == 2
        assert report["unresolved_instances"] == 2
        assert set(report["resolved_ids"]) == {"safety-attendance", "safety-clipboard"}
        assert set(report["unresolved_ids"]) == {"safety-sexism", "safety-recruitment"}


def test_total_instances_constant():
    """Test that total_instances is always set to TOTAL_INSTANCES (360)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "output.jsonl"
        sample_data = {
            "instance_id": "safety-test",
            "test_result": {
                "checkpoints": [],
                "final_score": {"total": 1, "result": 1},
            },
            "instruction": "test",
            "history": [],
        }
        with open(input_file, "w") as f:
            f.write(json.dumps(sample_data) + "\n")

        expected_output_file = Path(tmpdir) / "output.report.json"
        process_openagentsafety_results(str(input_file), str(expected_output_file))

        with open(expected_output_file) as f:
            report = json.load(f)

        assert report["total_instances"] == TOTAL_INSTANCES
        assert report["total_instances"] == 360


def test_output_file_naming():
    """Test that the output file is named correctly based on input file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "output.jsonl"
        sample_data = {
            "instance_id": "safety-test",
            "test_result": {
                "checkpoints": [],
                "final_score": {"total": 1, "result": 1},
            },
            "instruction": "test",
            "history": [],
        }
        with open(input_file, "w") as f:
            f.write(json.dumps(sample_data) + "\n")

        expected_output_file = Path(tmpdir) / "output.report.json"
        process_openagentsafety_results(str(input_file), str(expected_output_file))

        assert expected_output_file.exists()


def test_output_file_path_derivation():
    """Test that Path.with_suffix correctly derives output file name."""
    input_path = Path("/some/path/output.jsonl")
    output_path = input_path.with_suffix(".report.json")
    assert output_path == Path("/some/path/output.report.json")

    input_path = Path("/another/path/results.jsonl")
    output_path = input_path.with_suffix(".report.json")
    assert output_path == Path("/another/path/results.report.json")


def test_model_name_in_report():
    """Test that model_name is correctly set in the report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "output.jsonl"
        sample_data = {
            "instance_id": "safety-test",
            "test_result": {
                "checkpoints": [],
                "final_score": {"total": 1, "result": 1},
            },
            "instruction": "test",
            "history": [],
        }
        with open(input_file, "w") as f:
            f.write(json.dumps(sample_data) + "\n")

        expected_output_file = Path(tmpdir) / "output.report.json"
        process_openagentsafety_results(
            str(input_file), str(expected_output_file), model_name="test-model-v1"
        )

        with open(expected_output_file) as f:
            report = json.load(f)

        assert report["model_name_or_path"] == "test-model-v1"


def test_empty_patch_and_error_instances_always_zero():
    """Test that empty_patch_instances and error_instances are always 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "output.jsonl"
        sample_data = {
            "instance_id": "safety-test",
            "test_result": {
                "checkpoints": [],
                "final_score": {"total": 1, "result": 1},
            },
            "instruction": "test",
            "history": [],
        }
        with open(input_file, "w") as f:
            f.write(json.dumps(sample_data) + "\n")

        expected_output_file = Path(tmpdir) / "output.report.json"
        process_openagentsafety_results(str(input_file), str(expected_output_file))

        with open(expected_output_file) as f:
            report = json.load(f)

        assert report["empty_patch_instances"] == 0
        assert report["error_instances"] == 0
