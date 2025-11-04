"""
Test to verify that LLM metrics are collected properly in evaluations.

This test verifies the core pattern used in the fixed run_infer.py files:
collecting metrics from llm.metrics and including them in EvalOutput.
"""

import json

from pydantic import SecretStr

from benchmarks.utils.models import EvalOutput
from openhands.sdk import LLM


def test_metrics_collection_pattern():
    """
    Test the pattern for collecting metrics from LLM.
    
    This verifies that the pattern used in the fixed run_infer.py files
    (checking llm.metrics and calling model_dump()) works correctly.
    """
    # Create LLM - metrics are automatically initialized
    llm = LLM(
        usage_id="test",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test.com",
    )

    # This is the pattern used in the fixed run_infer.py files
    metrics = None
    if llm.metrics:
        metrics = llm.metrics.model_dump()

    # Verify metrics were collected (initialized with zeros by default)
    assert metrics is not None, "Metrics should be initialized and not None"
    assert "accumulated_cost" in metrics, "Should have accumulated_cost"
    assert "accumulated_token_usage" in metrics, "Should have accumulated_token_usage"
    assert isinstance(metrics["accumulated_cost"], (int, float))
    assert isinstance(metrics["accumulated_token_usage"], dict)

    # Create EvalOutput with metrics
    output = EvalOutput(
        instance_id="test-1",
        test_result={"result": "success"},
        instruction="test instruction",
        error=None,
        history=[],
        metrics=metrics,
    )

    # Verify the output can be serialized properly
    output_json = output.model_dump_json()
    output_data = json.loads(output_json)

    assert "metrics" in output_data, "Output should contain metrics field"
    assert output_data["metrics"] is not None, "Metrics should not be None in JSON"
    assert "accumulated_cost" in output_data["metrics"]
    assert "accumulated_token_usage" in output_data["metrics"]

    print("✓ Metrics collection pattern test passed")


def test_eval_output_with_no_metrics():
    """Test that EvalOutput can handle None metrics."""
    # Create EvalOutput without metrics
    output = EvalOutput(
        instance_id="test-2",
        test_result={},
        instruction="test",
        error=None,
        history=[],
        metrics=None,
    )

    # Verify the output can be serialized properly
    output_json = output.model_dump_json()
    output_data = json.loads(output_json)

    assert "metrics" in output_data, "Output should contain metrics field"
    assert output_data["metrics"] is None, "Metrics should be None in JSON"

    print("✓ EvalOutput with no metrics test passed")


def test_metrics_serialization():
    """Test that metrics can be properly serialized to JSON."""
    llm = LLM(
        usage_id="test",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test.com",
    )

    # Get metrics using the pattern from fixed files
    metrics = None
    if llm.metrics:
        metrics = llm.metrics.model_dump()

    # Ensure it can be JSON serialized
    metrics_json = json.dumps(metrics)
    assert metrics_json is not None
    
    # Ensure it can be deserialized
    metrics_dict = json.loads(metrics_json)
    assert "accumulated_cost" in metrics_dict
    assert "accumulated_token_usage" in metrics_dict

    print("✓ Metrics serialization test passed")
