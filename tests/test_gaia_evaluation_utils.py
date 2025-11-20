"""Tests for GAIA evaluation utils integration.

This test verifies that GAIA correctly uses the shared evaluation_utils
functions instead of local implementations.
"""

import tempfile
from pathlib import Path

from benchmarks.utils.evaluation_utils import get_default_on_result_writer
from benchmarks.utils.models import EvalInstance, EvalOutput
from openhands.sdk.llm import Metrics


def test_gaia_imports_get_default_on_result_writer():
    """Test that GAIA run_infer.py imports get_default_on_result_writer from evaluation_utils."""
    # Import the GAIA module
    from benchmarks.gaia import run_infer

    # Check that get_default_on_result_writer is available in the module
    assert hasattr(run_infer, "get_default_on_result_writer")

    # Verify it's the same function from evaluation_utils
    assert run_infer.get_default_on_result_writer is get_default_on_result_writer


def test_get_default_on_result_writer_functionality():
    """Test that get_default_on_result_writer works correctly."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as f:
        output_path = f.name

    try:
        # Create the callback function
        callback = get_default_on_result_writer(output_path)

        # Create test data
        instance = EvalInstance(id="test_instance", data={"test": "data"})
        output = EvalOutput(
            instance_id="test_instance",
            test_result={"score": 1.0},
            instruction="test instruction",
            error=None,
            history=[],
            metrics=Metrics(),
            instance={"test": "data"},
        )

        # Call the callback
        callback(instance, output)

        # Verify the output was written
        with open(output_path, "r") as f:
            content = f.read().strip()
            assert content  # Should have content

            # Parse the JSON to verify it's valid
            import json

            parsed = json.loads(content)
            assert parsed["instance_id"] == "test_instance"
            assert parsed["test_result"]["score"] == 1.0

    finally:
        # Clean up
        Path(output_path).unlink(missing_ok=True)


def test_get_default_on_result_writer_skips_errors():
    """Test that get_default_on_result_writer skips outputs with errors."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as f:
        output_path = f.name

    try:
        # Create the callback function
        callback = get_default_on_result_writer(output_path)

        # Create test data with error
        instance = EvalInstance(id="test_instance", data={"test": "data"})
        output = EvalOutput(
            instance_id="test_instance",
            test_result={"score": 0.0},
            instruction="test instruction",
            error="Test error",  # This should cause the output to be skipped
            history=[],
            metrics=Metrics(),
            instance={"test": "data"},
        )

        # Call the callback
        callback(instance, output)

        # Verify nothing was written
        with open(output_path, "r") as f:
            content = f.read().strip()
            assert content == ""  # Should be empty

    finally:
        # Clean up
        Path(output_path).unlink(missing_ok=True)
