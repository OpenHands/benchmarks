"""Workspace cleanup behavior in the evaluation orchestrator."""

from unittest.mock import Mock

from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import LLM
from openhands.sdk.critic import PassCritic


def _metadata(eval_dir: str) -> EvalMetadata:
    llm = LLM(model="test-model")
    return EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir=eval_dir,
        details={},
        eval_limit=1,
        max_attempts=1,
        max_retries=0,
        critic=PassCritic(),
    )


def test_workspace_cleanup_called_on_success(tmp_path) -> None:
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

    class TestEvaluation(Evaluation):
        def prepare_instances(self):
            return [test_instance]

        def prepare_workspace(self, instance: EvalInstance):
            return mock_workspace

        def evaluate_instance(self, instance, workspace, attempt: int):
            return test_output

    metadata = _metadata(str(tmp_path)).model_copy(update={"max_retries": 2})
    evaluator = TestEvaluation(metadata=metadata, num_workers=1)
    result_instance, result_output = evaluator._process_one_mp(
        test_instance, attempt=1
    )

    assert mock_workspace.__exit__.call_count == 1
    mock_workspace.__exit__.assert_called_with(None, None, None)
    assert result_instance.id == "test_instance"
    assert result_output.error is None


def test_workspace_cleanup_called_on_failure(tmp_path) -> None:
    mock_workspace = Mock()
    mock_workspace.__exit__ = Mock()
    test_instance = EvalInstance(id="test_instance", data={"test": "data"})

    class TestEvaluation(Evaluation):
        def prepare_instances(self):
            return [test_instance]

        def prepare_workspace(self, instance: EvalInstance):
            return mock_workspace

        def evaluate_instance(self, instance, workspace, attempt: int):
            raise RuntimeError("Test evaluation failure")

    metadata = _metadata(str(tmp_path)).model_copy(update={"max_retries": 2})
    evaluator = TestEvaluation(metadata=metadata, num_workers=1)
    result_instance, result_output = evaluator._process_one_mp(
        test_instance, attempt=1
    )

    assert mock_workspace.__exit__.call_count == metadata.max_retries + 1
    mock_workspace.__exit__.assert_called_with(None, None, None)
    assert result_instance.id == "test_instance"
    assert result_output.error is not None


def test_workspace_cleanup_handles_cleanup_exception(tmp_path) -> None:
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

    class TestEvaluation(Evaluation):
        def prepare_instances(self):
            return [test_instance]

        def prepare_workspace(self, instance: EvalInstance):
            return mock_workspace

        def evaluate_instance(self, instance, workspace, attempt: int):
            return test_output

    metadata = _metadata(str(tmp_path)).model_copy(update={"max_retries": 2})
    evaluator = TestEvaluation(metadata=metadata, num_workers=1)
    result_instance, result_output = evaluator._process_one_mp(
        test_instance, attempt=1
    )

    mock_workspace.__exit__.assert_called_once()
    assert result_instance.id == "test_instance"
    assert result_output.error is None


def test_workspace_cleanup_with_retries(tmp_path) -> None:
    workspaces_created = []

    def create_mock_workspace():
        workspace = Mock()
        workspace.__exit__ = Mock()
        workspaces_created.append(workspace)
        return workspace

    test_instance = EvalInstance(id="test_instance", data={"test": "data"})
    attempt_count = 0

    class TestEvaluation(Evaluation):
        def prepare_instances(self):
            return [test_instance]

        def prepare_workspace(self, instance: EvalInstance):
            return create_mock_workspace()

        def evaluate_instance(self, instance, workspace, attempt: int):
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

    metadata = _metadata(str(tmp_path)).model_copy(update={"max_retries": 2})
    evaluator = TestEvaluation(metadata=metadata, num_workers=1)
    result_instance, result_output = evaluator._process_one_mp(
        test_instance, attempt=1
    )

    assert len(workspaces_created) == metadata.max_retries + 1
    for workspace in workspaces_created:
        workspace.__exit__.assert_called_once_with(None, None, None)

    assert result_instance.id == "test_instance"
    assert result_output.error is None
