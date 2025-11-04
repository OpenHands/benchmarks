"""Integration tests for metrics collection in evaluation workflows.

These tests call the actual evaluate_instance methods from the evaluation
classes with mocked dependencies to verify that metrics are properly collected
from the LLM and included in the EvalOutput.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import LLM
from openhands.sdk.llm.utils.metrics import Metrics, TokenUsage
from openhands.sdk.workspace import RemoteWorkspace


@pytest.fixture
def mock_llm_with_metrics():
    """Create a mock LLM with populated metrics."""
    llm = LLM(model="test-model")

    # Create realistic metrics
    metrics = Metrics(
        model_name="test-model",
        accumulated_cost=1.5,
        accumulated_token_usage=TokenUsage(
            model="test-model",
            prompt_tokens=100,
            completion_tokens=50,
        ),
    )

    # Assign metrics to the LLM using restore_metrics
    llm.restore_metrics(metrics)
    return llm


@pytest.fixture
def mock_workspace():
    """Create a mock workspace."""
    workspace = MagicMock(spec=RemoteWorkspace)
    workspace.working_dir = "/workspace"
    workspace.execute_command = MagicMock(
        return_value=MagicMock(
            exit_code=0,
            stdout="test output",
            stderr="",
        )
    )
    return workspace


@pytest.fixture
def mock_conversation():
    """Create a mock conversation that simulates a conversation run."""
    conversation = MagicMock()
    conversation.state.events = []
    return conversation


def test_swe_bench_metrics_collection(mock_llm_with_metrics, mock_workspace):
    """Test that SWE-Bench evaluation collects metrics from LLM."""
    from benchmarks.swe_bench.run_infer import SWEBenchEvaluation

    # Create test instance
    instance = EvalInstance(
        id="test__instance-1",
        data={
            "repo": "test/repo",
            "instance_id": "test__instance-1",
            "base_commit": "abc123",
            "problem_statement": "Test problem",
            "hints_text": "",
            "created_at": "2024-01-01",
            "patch": "test patch",
            "test_patch": "test test_patch",
            "version": "1.0",
            "FAIL_TO_PASS": '["test1"]',
            "PASS_TO_PASS": '["test2"]',
            "environment_setup_commit": "abc123",
        },
    )

    # Create metadata with mocked LLM
    metadata = EvalMetadata(
        dataset="princeton-nlp/SWE-bench_Lite",
        dataset_split="test",
        llm=mock_llm_with_metrics,
        max_iterations=5,
        eval_output_dir="/tmp/eval_output",
        critic_name="test_critic",
    )

    # Create evaluation instance
    evaluation = SWEBenchEvaluation(metadata=metadata)

    # Mock the conversation to avoid actual LLM calls
    with (
        patch("benchmarks.swe_bench.run_infer.Conversation") as mock_conv_class,
        patch("benchmarks.swe_bench.run_infer.Agent"),
        patch("benchmarks.swe_bench.run_infer.get_default_tools"),
        patch("benchmarks.swe_bench.run_infer.get_instruction") as mock_get_instruction,
    ):
        # Setup mocks
        mock_get_instruction.return_value = "Test instruction"
        mock_conversation = MagicMock()
        mock_conversation.state.events = []
        mock_conv_class.return_value = mock_conversation

        # Call the actual evaluate_instance method
        result = evaluation.evaluate_instance(instance, mock_workspace)

    # Verify result is EvalOutput
    assert isinstance(result, EvalOutput)

    # Verify metrics were collected
    assert result.metrics is not None
    assert "accumulated_cost" in result.metrics
    assert result.metrics["accumulated_cost"] == 1.5
    assert "accumulated_token_usage" in result.metrics
    assert result.metrics["accumulated_token_usage"]["prompt_tokens"] == 100
    assert result.metrics["accumulated_token_usage"]["completion_tokens"] == 50

    # Verify metrics can be serialized to JSON
    json_str = json.dumps(result.model_dump())
    assert json_str is not None
    parsed = json.loads(json_str)
    assert parsed["metrics"]["accumulated_cost"] == 1.5

    print("✓ SWE-Bench integration test passed")


def test_gaia_metrics_collection(mock_llm_with_metrics, mock_workspace):
    """Test that GAIA evaluation collects metrics from LLM."""
    from benchmarks.gaia.run_infer import GAIAEvaluation

    # Create test instance
    instance = EvalInstance(
        id="test-instance-1",
        data={
            "task_id": "test-instance-1",
            "Question": "What is the answer?",
            "Level": 1,
            "Final answer": "42",
            "file_name": "",
            "Annotator Metadata": '{"test": true}',
        },
    )

    # Create metadata with mocked LLM
    metadata = EvalMetadata(
        dataset="gaia-benchmark/GAIA",
        dataset_split="test",
        llm=mock_llm_with_metrics,
        max_iterations=5,
        eval_output_dir="/tmp/eval_output",
        critic_name="test_critic",
        details={"test": True},
    )

    # Create evaluation instance
    evaluation = GAIAEvaluation(metadata=metadata)

    # Mock the conversation to avoid actual LLM calls
    with (
        patch("benchmarks.gaia.run_infer.Conversation") as mock_conv_class,
        patch("benchmarks.gaia.run_infer.Agent"),
        patch("benchmarks.gaia.run_infer.get_default_tools"),
        patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
    ):
        # Setup mocks - simulate a conversation with an answer
        mock_conversation = MagicMock()
        mock_event = MagicMock()
        mock_event.model_dump.return_value = {
            "type": "agent",
            "content": "<solution>42</solution>",
        }
        mock_conversation.state.events = [mock_event]
        mock_conv_class.return_value = mock_conversation

        # Call the actual evaluate_instance method
        result = evaluation.evaluate_instance(instance, mock_workspace)

    # Verify result is EvalOutput
    assert isinstance(result, EvalOutput)

    # Verify metrics were collected
    assert result.metrics is not None
    assert "accumulated_cost" in result.metrics
    assert result.metrics["accumulated_cost"] == 1.5
    assert "accumulated_token_usage" in result.metrics
    assert result.metrics["accumulated_token_usage"]["prompt_tokens"] == 100
    assert result.metrics["accumulated_token_usage"]["completion_tokens"] == 50

    # Verify metrics can be serialized to JSON
    json_str = json.dumps(result.model_dump())
    assert json_str is not None
    parsed = json.loads(json_str)
    assert parsed["metrics"]["accumulated_cost"] == 1.5

    # Verify test result includes score
    assert "score" in result.test_result

    print("✓ GAIA integration test passed")


def test_metrics_with_zero_cost(mock_workspace):
    """Test that metrics are collected even when cost is zero."""
    from benchmarks.swe_bench.run_infer import SWEBenchEvaluation

    # Create LLM with default metrics (cost = 0)
    llm = LLM(model="test-model")
    # LLM metrics are initialized by default with zero cost

    # Create test instance
    instance = EvalInstance(
        id="test__instance-2",
        data={
            "repo": "test/repo",
            "instance_id": "test__instance-2",
            "base_commit": "abc123",
            "problem_statement": "Test problem",
            "hints_text": "",
            "created_at": "2024-01-01",
            "patch": "test patch",
            "test_patch": "test test_patch",
            "version": "1.0",
            "FAIL_TO_PASS": '["test1"]',
            "PASS_TO_PASS": '["test2"]',
            "environment_setup_commit": "abc123",
        },
    )

    # Create metadata with LLM that has zero-cost metrics
    metadata = EvalMetadata(
        dataset="princeton-nlp/SWE-bench_Lite",
        dataset_split="test",
        llm=llm,
        max_iterations=5,
        eval_output_dir="/tmp/eval_output",
        critic_name="test_critic",
    )

    # Create evaluation instance
    evaluation = SWEBenchEvaluation(metadata=metadata)

    # Mock the conversation to avoid actual LLM calls
    with (
        patch("benchmarks.swe_bench.run_infer.Conversation") as mock_conv_class,
        patch("benchmarks.swe_bench.run_infer.Agent"),
        patch("benchmarks.swe_bench.run_infer.get_default_tools"),
        patch("benchmarks.swe_bench.run_infer.get_instruction") as mock_get_instruction,
    ):
        # Setup mocks
        mock_get_instruction.return_value = "Test instruction"
        mock_conversation = MagicMock()
        mock_conversation.state.events = []
        mock_conv_class.return_value = mock_conversation

        # Call the actual evaluate_instance method
        result = evaluation.evaluate_instance(instance, mock_workspace)

    # Verify result is EvalOutput
    assert isinstance(result, EvalOutput)

    # Verify metrics are collected even with zero cost
    assert result.metrics is not None
    assert "accumulated_cost" in result.metrics
    assert result.metrics["accumulated_cost"] == 0.0

    # Verify it can still be serialized to JSON
    json_str = json.dumps(result.model_dump())
    assert json_str is not None

    print("✓ Zero-cost metrics test passed")
