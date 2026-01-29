"""Tests for workspace cleanup functionality in the evaluation module."""

from typing import List
from unittest.mock import Mock

import pytest

from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import LLM
from openhands.sdk.critic import PassCritic


def test_workspace_cleanup_called_on_success():
    """Test that workspace cleanup is called when evaluation succeeds."""
    from benchmarks.utils.evaluation import Evaluation

    mock_workspace = Mock()
    mock_workspace.__exit__ = Mock()

    test_instance = EvalInstance(id="test_instance", data={"test": "data"})
    test_output = EvalOutput(
        instance_id="test_instance",
        test_result={"success": True},
        instruction="test instruction",
        error=None,
        history=[],
        instance={"test": "data"},
    )

    llm = LLM(model="test-model")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir="/tmp/test",
        details={},
        eval_limit=1,
        n_critic_runs=1,
        max_retries=0,
        critic=PassCritic(),
    )

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
    result_instance, result_output = evaluator._process_one_mp(
        test_instance, None, critic_attempt=1
    )

    mock_workspace.__exit__.assert_called_once_with(None, None, None)
    assert result_instance.id == "test_instance"
    assert result_output.instance_id == "test_instance"
    assert result_output.error is None


def test_workspace_cleanup_called_on_failure():
    """Test that workspace cleanup is called when evaluation fails."""
    from benchmarks.utils.evaluation import Evaluation

    mock_workspace = Mock()
    mock_workspace.__exit__ = Mock()

    test_instance = EvalInstance(id="test_instance", data={"test": "data"})

    llm = LLM(model="test-model")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir="/tmp/test",
        details={},
        eval_limit=1,
        n_critic_runs=1,
        max_retries=0,
        critic=PassCritic(),
    )

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
    result_instance, result_output = evaluator._process_one_mp(
        test_instance, None, critic_attempt=1
    )

    mock_workspace.__exit__.assert_called_once_with(None, None, None)
    assert result_instance.id == "test_instance"
    assert result_output.instance_id == "test_instance"
    assert result_output.error is not None
    assert "Test evaluation failure" in result_output.error


def test_workspace_cleanup_handles_cleanup_exception():
    """Test that evaluation continues even if workspace cleanup fails."""
    from benchmarks.utils.evaluation import Evaluation

    mock_workspace = Mock()
    mock_workspace.__exit__ = Mock(side_effect=RuntimeError("Cleanup failed"))

    test_instance = EvalInstance(id="test_instance", data={"test": "data"})
    test_output = EvalOutput(
        instance_id="test_instance",
        test_result={"success": True},
        instruction="test instruction",
        error=None,
        history=[],
        instance={"test": "data"},
    )

    llm = LLM(model="test-model")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir="/tmp/test",
        details={},
        eval_limit=1,
        n_critic_runs=1,
        max_retries=0,
        critic=PassCritic(),
    )

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
    result_instance, result_output = evaluator._process_one_mp(
        test_instance, None, critic_attempt=1
    )

    mock_workspace.__exit__.assert_called_once_with(None, None, None)
    assert result_instance.id == "test_instance"
    assert result_output.instance_id == "test_instance"
    assert result_output.error is None


def test_workspace_cleanup_with_retries():
    """Test that workspace cleanup is called for each retry attempt."""
    from benchmarks.utils.evaluation import Evaluation

    workspaces_created = []

    def create_mock_workspace():
        workspace = Mock()
        workspace.__exit__ = Mock()
        workspaces_created.append(workspace)
        return workspace

    test_instance = EvalInstance(id="test_instance", data={"test": "data"})

    llm = LLM(model="test-model")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir="/tmp/test",
        details={},
        eval_limit=1,
        n_critic_runs=1,
        max_retries=2,
        critic=PassCritic(),
    )

    attempt_count = 0

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
    result_instance, result_output = evaluator._process_one_mp(
        test_instance, None, critic_attempt=1
    )

    assert len(workspaces_created) == 3
    for workspace in workspaces_created:
        workspace.__exit__.assert_called_once_with(None, None, None)

    assert result_instance.id == "test_instance"
    assert result_output.instance_id == "test_instance"
    assert result_output.error is None


if __name__ == "__main__":
    pytest.main([__file__])
