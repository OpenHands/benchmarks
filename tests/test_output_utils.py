"""Tests for output_utils module."""

import json
import tempfile
from pathlib import Path

import pytest

from benchmarks.utils.output_utils import add_resolve_rate_to_predictions


@pytest.fixture
def sample_predictions():
    """Sample predictions data."""
    return [
        {"instance_id": "fasterxml/jackson-databind:pr-4469", "model_patch": "patch1"},
        {"instance_id": "elastic/logstash:pr-15241", "model_patch": "patch2"},
        {"instance_id": "fasterxml/jackson-databind:pr-1234", "model_patch": "patch3"},
        {"instance_id": "fasterxml/jackson-databind:pr-2036", "model_patch": "patch4"},
    ]


@pytest.fixture
def sample_report():
    """Sample report data."""
    return {
        "total_instances": 9,
        "submitted_instances": 9,
        "completed_instances": 9,
        "resolved_instances": 3,
        "unresolved_instances": 6,
        "resolved_ids": [
            "fasterxml/jackson-databind:pr-2036",
            "fasterxml/jackson-core:pr-1016",
            "fasterxml/jackson-databind:pr-4228",
        ],
        "unresolved_ids": [
            "fasterxml/jackson-core:pr-964",
            "fasterxml/jackson-databind:pr-4469",
            "elastic/logstash:pr-13997",
            "elastic/logstash:pr-16079",
            "elastic/logstash:pr-15241",
            "elastic/logstash:pr-14970",
        ],
    }


def test_add_resolve_rate_to_predictions(sample_predictions, sample_report):
    """Test that resolution status is correctly added to predictions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        predictions_path = Path(tmpdir) / "predictions.jsonl"
        report_path = Path(tmpdir) / "report.json"

        # Write predictions
        with open(predictions_path, "w") as f:
            for pred in sample_predictions:
                f.write(json.dumps(pred) + "\n")

        # Write report
        with open(report_path, "w") as f:
            json.dump(sample_report, f)

        # Run the function
        add_resolve_rate_to_predictions(predictions_path, report_path)

        # Read updated predictions
        updated_predictions = []
        with open(predictions_path, "r") as f:
            for line in f:
                updated_predictions.append(json.loads(line))

        # Verify results
        assert len(updated_predictions) == 4

        # pr-4469 should be unresolved
        pred_4469 = next(
            p
            for p in updated_predictions
            if p["instance_id"] == "fasterxml/jackson-databind:pr-4469"
        )
        assert pred_4469["report"] == {"resolved": False}

        # pr-15241 should be unresolved
        pred_15241 = next(
            p
            for p in updated_predictions
            if p["instance_id"] == "elastic/logstash:pr-15241"
        )
        assert pred_15241["report"] == {"resolved": False}

        # pr-1234 should have no report (not in either list)
        pred_1234 = next(
            p
            for p in updated_predictions
            if p["instance_id"] == "fasterxml/jackson-databind:pr-1234"
        )
        assert "report" not in pred_1234

        # pr-2036 should be resolved
        pred_2036 = next(
            p
            for p in updated_predictions
            if p["instance_id"] == "fasterxml/jackson-databind:pr-2036"
        )
        assert pred_2036["report"] == {"resolved": True}


def test_add_resolve_rate_preserves_other_fields(sample_report):
    """Test that other fields in predictions are preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        predictions_path = Path(tmpdir) / "predictions.jsonl"
        report_path = Path(tmpdir) / "report.json"

        # Write prediction with extra fields
        prediction = {
            "instance_id": "fasterxml/jackson-databind:pr-2036",
            "model_patch": "some patch",
            "extra_field": "extra_value",
            "nested": {"key": "value"},
        }
        with open(predictions_path, "w") as f:
            f.write(json.dumps(prediction) + "\n")

        # Write report
        with open(report_path, "w") as f:
            json.dump(sample_report, f)

        # Run the function
        add_resolve_rate_to_predictions(predictions_path, report_path)

        # Read updated prediction
        with open(predictions_path, "r") as f:
            updated = json.loads(f.readline())

        # Verify all original fields are preserved
        assert updated["instance_id"] == "fasterxml/jackson-databind:pr-2036"
        assert updated["model_patch"] == "some patch"
        assert updated["extra_field"] == "extra_value"
        assert updated["nested"] == {"key": "value"}
        assert updated["report"] == {"resolved": True}


def test_add_resolve_rate_empty_predictions(sample_report):
    """Test handling of empty predictions file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        predictions_path = Path(tmpdir) / "predictions.jsonl"
        report_path = Path(tmpdir) / "report.json"

        # Write empty predictions file
        with open(predictions_path, "w") as f:
            pass

        # Write report
        with open(report_path, "w") as f:
            json.dump(sample_report, f)

        # Run the function
        add_resolve_rate_to_predictions(predictions_path, report_path)

        # Read updated predictions
        with open(predictions_path, "r") as f:
            content = f.read()

        assert content == ""


def test_add_resolve_rate_empty_report_lists():
    """Test handling of report with empty resolved/unresolved lists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        predictions_path = Path(tmpdir) / "predictions.jsonl"
        report_path = Path(tmpdir) / "report.json"

        # Write predictions
        prediction = {"instance_id": "test-instance", "data": "test"}
        with open(predictions_path, "w") as f:
            f.write(json.dumps(prediction) + "\n")

        # Write report with empty lists
        report = {"resolved_ids": [], "unresolved_ids": []}
        with open(report_path, "w") as f:
            json.dump(report, f)

        # Run the function
        add_resolve_rate_to_predictions(predictions_path, report_path)

        # Read updated prediction
        with open(predictions_path, "r") as f:
            updated = json.loads(f.readline())

        # No report should be added since instance is not in either list
        assert "report" not in updated


def test_add_resolve_rate_with_string_paths(sample_predictions, sample_report):
    """Test that function works with string paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        predictions_path = str(Path(tmpdir) / "predictions.jsonl")
        report_path = str(Path(tmpdir) / "report.json")

        # Write predictions
        with open(predictions_path, "w") as f:
            for pred in sample_predictions:
                f.write(json.dumps(pred) + "\n")

        # Write report
        with open(report_path, "w") as f:
            json.dump(sample_report, f)

        # Run the function with string paths
        add_resolve_rate_to_predictions(predictions_path, report_path)

        # Read updated predictions
        with open(predictions_path, "r") as f:
            updated = json.loads(f.readline())

        # Verify it worked
        assert updated["report"] == {"resolved": False}


def test_add_resolve_rate_missing_keys_in_report():
    """Test handling of report missing resolved_ids or unresolved_ids keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        predictions_path = Path(tmpdir) / "predictions.jsonl"
        report_path = Path(tmpdir) / "report.json"

        # Write predictions
        prediction = {"instance_id": "test-instance", "data": "test"}
        with open(predictions_path, "w") as f:
            f.write(json.dumps(prediction) + "\n")

        # Write report without resolved_ids and unresolved_ids
        report = {"total_instances": 1}
        with open(report_path, "w") as f:
            json.dump(report, f)

        # Run the function - should not raise
        add_resolve_rate_to_predictions(predictions_path, report_path)

        # Read updated prediction
        with open(predictions_path, "r") as f:
            updated = json.loads(f.readline())

        # No report should be added
        assert "report" not in updated
