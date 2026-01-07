"""Tests for workspace cleanup functionality in the evaluation module."""

from typing import List
from unittest.mock import Mock

import pytest

from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import LLM
from openhands.sdk.critic import PassCritic


def test_workspace_cleanup_called_on_success():
    """Test that workspace cleanup is called when evaluation succeeds."""
    # Import here to avoid circular imports
    from benchmarks.utils.evaluation import Evaluation

    # Create a mock workspace
    mock_workspace = Mock()
    mock_workspace.__exit__ = Mock()

    # Create test instance
    test_instance = EvalInstance(id="test_instance", data={"test": "data"})

    # Create test output
    test_output = EvalOutput(
        instance_id="test_instance",
        test_result={"success": True},
        instruction="test instruction",
        error=None,
        history=[],
        instance={"test": "data"},
    )

    # Create evaluation metadata
    llm = LLM(model="test-model")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir="/tmp/test",
        details={},
        eval_limit=1,
        max_attempts=1,
        max_retries=0,
        critic=PassCritic(),
    )

    # Create a concrete evaluation class for testing
    class TestEvaluation(Evaluation):
        def prepare_instances(self) -> List[EvalInstance]:
            return [test_instance]

        def prepare_workspace(
            self,
            instance: EvalInstance,
            resource_factor: int = 1,
            forward_env: list[str] | None = None,
        ):
            mock_workspace.forward_env = forward_env or []
            mock_workspace.resource_factor = resource_factor
            return mock_workspace

        def evaluate_instance(self, instance, workspace):
            return test_output

    evaluator = TestEvaluation(metadata=metadata, num_workers=1)

    # Call the method directly
    result_instance, result_output = evaluator._process_one_mp(test_instance, None)

    # Verify the workspace cleanup was called
    mock_workspace.__exit__.assert_called_once_with(None, None, None)
    assert result_instance.id == "test_instance"
    assert result_output.instance_id == "test_instance"
    assert result_output.error is None


def test_workspace_cleanup_called_on_failure():
    """Test that workspace cleanup is called when evaluation fails."""
    # Import here to avoid circular imports
    from benchmarks.utils.evaluation import Evaluation

    # Create a mock workspace
    mock_workspace = Mock()
    mock_workspace.__exit__ = Mock()

    # Create test instance
    test_instance = EvalInstance(id="test_instance", data={"test": "data"})

    # Create evaluation metadata
    llm = LLM(model="test-model")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir="/tmp/test",
        details={},
        eval_limit=1,
        max_attempts=1,
        max_retries=0,
        critic=PassCritic(),
    )

    # Create a concrete evaluation class for testing
    class TestEvaluation(Evaluation):
        def prepare_instances(self) -> List[EvalInstance]:
            return [test_instance]

        def prepare_workspace(
            self,
            instance: EvalInstance,
            resource_factor: int = 1,
            forward_env: list[str] | None = None,
        ):
            mock_workspace.forward_env = forward_env or []
            mock_workspace.resource_factor = resource_factor
            return mock_workspace

        def evaluate_instance(self, instance, workspace):
            raise RuntimeError("Test evaluation failure")

    evaluator = TestEvaluation(metadata=metadata, num_workers=1)

    # Call the method directly
    result_instance, result_output = evaluator._process_one_mp(test_instance, None)

    # Verify the workspace cleanup was called even on failure
    mock_workspace.__exit__.assert_called_once_with(None, None, None)
    assert result_instance.id == "test_instance"
    assert result_output.instance_id == "test_instance"
    assert result_output.error is not None
    assert "Test evaluation failure" in result_output.error


def test_workspace_cleanup_handles_cleanup_exception():
    """Test that evaluation continues even if workspace cleanup fails."""
    # Import here to avoid circular imports
    from benchmarks.utils.evaluation import Evaluation

    # Create a mock workspace that fails on cleanup
    mock_workspace = Mock()
    mock_workspace.__exit__ = Mock(side_effect=RuntimeError("Cleanup failed"))

    # Create test instance
    test_instance = EvalInstance(id="test_instance", data={"test": "data"})

    # Create test output
    test_output = EvalOutput(
        instance_id="test_instance",
        test_result={"success": True},
        instruction="test instruction",
        error=None,
        history=[],
        instance={"test": "data"},
    )

    # Create evaluation metadata
    llm = LLM(model="test-model")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir="/tmp/test",
        details={},
        eval_limit=1,
        max_attempts=1,
        max_retries=0,
        critic=PassCritic(),
    )

    # Create a concrete evaluation class for testing
    class TestEvaluation(Evaluation):
        def prepare_instances(self) -> List[EvalInstance]:
            return [test_instance]

        def prepare_workspace(
            self,
            instance: EvalInstance,
            resource_factor: int = 1,
            forward_env: list[str] | None = None,
        ):
            mock_workspace.forward_env = forward_env or []
            mock_workspace.resource_factor = resource_factor
            return mock_workspace

        def evaluate_instance(self, instance, workspace):
            return test_output

    evaluator = TestEvaluation(metadata=metadata, num_workers=1)

    # Call the method directly - should not raise an exception
    result_instance, result_output = evaluator._process_one_mp(test_instance, None)

    # Verify the workspace cleanup was attempted
    mock_workspace.__exit__.assert_called_once_with(None, None, None)
    assert result_instance.id == "test_instance"
    assert result_output.instance_id == "test_instance"
    assert result_output.error is None  # Main evaluation should still succeed


def test_workspace_cleanup_with_retries():
    """Test that workspace cleanup is called for each retry attempt."""
    # Import here to avoid circular imports
    from benchmarks.utils.evaluation import Evaluation

    # Track all workspaces created
    workspaces_created = []

    def create_mock_workspace():
        workspace = Mock()
        workspace.__exit__ = Mock()
        workspaces_created.append(workspace)
        return workspace

    # Create test instance
    test_instance = EvalInstance(id="test_instance", data={"test": "data"})

    # Create evaluation metadata with retries
    llm = LLM(model="test-model")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir="/tmp/test",
        details={},
        eval_limit=1,
        max_attempts=1,
        max_retries=2,  # Allow 2 retries
        critic=PassCritic(),
    )

    # Track evaluation attempts
    attempt_count = 0

    # Create a concrete evaluation class for testing
    class TestEvaluation(Evaluation):
        def prepare_instances(self) -> List[EvalInstance]:
            return [test_instance]

        def prepare_workspace(
            self,
            instance: EvalInstance,
            resource_factor: int = 1,
            forward_env: list[str] | None = None,
        ):
            workspace = create_mock_workspace()
            workspace.forward_env = forward_env or []
            workspace.resource_factor = resource_factor
            return workspace

        def evaluate_instance(self, instance, workspace):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count <= 2:
                raise RuntimeError(f"Attempt {attempt_count} failed")
            return EvalOutput(
                instance_id=instance.id,
                test_result={"success": True},
                instruction="test instruction",
                error=None,
                history=[],
                instance=instance.data,
            )

    evaluator = TestEvaluation(metadata=metadata, num_workers=1)

    # Call the method directly
    result_instance, result_output = evaluator._process_one_mp(test_instance, None)

    # Verify cleanup was called for all attempts (3 total: initial + 2 retries)
    assert len(workspaces_created) == 3, "Should create workspace for each attempt"
    for workspace in workspaces_created:
        workspace.__exit__.assert_called_once_with(None, None, None)

    # Final result should be successful
    assert result_instance.id == "test_instance"
    assert result_output.instance_id == "test_instance"
    assert result_output.error is None


def test_resource_factor_increases_on_runtime_failures():
    """Test that resource factor increases exponentially on runtime failures."""
    # Import here to avoid circular imports
    from benchmarks.utils.evaluation import Evaluation
    from benchmarks.utils.exceptions import AgentRuntimeDisconnectedError

    # Track resource factors passed to prepare_workspace
    resource_factors_used = []

    def create_mock_workspace(resource_factor):
        workspace = Mock()
        workspace.__exit__ = Mock()
        resource_factors_used.append(resource_factor)
        return workspace

    # Create test instance
    test_instance = EvalInstance(id="test_instance", data={"test": "data"})

    # Create evaluation metadata with retries
    llm = LLM(model="test-model")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir="/tmp/test",
        details={},
        eval_limit=1,
        max_attempts=1,
        max_retries=3,  # Allow 3 retries
        base_resource_factor=1,
        max_resource_factor=8,
        critic=PassCritic(),
    )

    # Track evaluation attempts
    attempt_count = 0

    # Create a concrete evaluation class for testing with resource_factor support
    class TestEvaluation(Evaluation):
        def prepare_instances(self) -> List[EvalInstance]:
            return [test_instance]

        def prepare_workspace(
            self,
            instance: EvalInstance,
            resource_factor: int = 1,
            forward_env: list[str] | None = None,
        ):
            workspace = create_mock_workspace(resource_factor)
            workspace.forward_env = forward_env or []
            return workspace

        def evaluate_instance(self, instance, workspace):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count <= 3:
                # Simulate runtime crashes with fatal runtime error
                raise AgentRuntimeDisconnectedError(
                    f"Runtime disconnected on attempt {attempt_count}"
                )
            return EvalOutput(
                instance_id=instance.id,
                test_result={"success": True},
                instruction="test instruction",
                error=None,
                history=[],
                instance=instance.data,
            )

    evaluator = TestEvaluation(metadata=metadata, num_workers=1)

    # Call the method directly
    result_instance, result_output = evaluator._process_one_mp(test_instance)

    # Verify resource factors increased exponentially: 1, 2, 4, 8
    assert len(resource_factors_used) == 4, "Should have 4 attempts"
    assert resource_factors_used[0] == 1, "First attempt should have factor 1"
    assert resource_factors_used[1] == 2, "Second attempt should have factor 2"
    assert resource_factors_used[2] == 4, "Third attempt should have factor 4"
    assert resource_factors_used[3] == 8, "Fourth attempt should have factor 8"

    # Final result should be successful
    assert result_instance.id == "test_instance"
    assert result_output.instance_id == "test_instance"
    assert result_output.error is None


if __name__ == "__main__":
    pytest.main([__file__])
