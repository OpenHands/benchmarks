"""Tests for proxy cost retrieval in the evaluation worker."""

from unittest.mock import Mock, call, patch

from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import LLM
from openhands.sdk.critic import PassCritic


def _make_metadata(tmp_path) -> EvalMetadata:
    return EvalMetadata(
        llm=LLM(model="test-model"),
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir=str(tmp_path),
        details={},
        eval_limit=1,
        n_critic_runs=1,
        max_retries=0,
        critic=PassCritic(),
    )


def _make_output(instance: EvalInstance) -> EvalOutput:
    return EvalOutput(
        instance_id=instance.id,
        test_result={},
        instruction="test instruction",
        error=None,
        history=[],
        instance=instance.data,
    )


def _build_evaluator(instance: EvalInstance, tmp_path):
    from benchmarks.utils.evaluation import Evaluation

    class TestEvaluation(Evaluation):
        def prepare_instances(self) -> list[EvalInstance]:
            return [instance]

        def prepare_workspace(
            self,
            instance: EvalInstance,
            resource_factor: int = 1,
            forward_env: list[str] | None = None,
        ):
            workspace = Mock()
            workspace.__exit__ = Mock()
            return workspace

        def evaluate_instance(self, instance, workspace):
            return _make_output(instance)

    return TestEvaluation(metadata=_make_metadata(tmp_path), num_workers=1)


def test_proxy_cost_retries_after_initial_zero_spend(tmp_path):
    instance = EvalInstance(id="test_instance", data={"test": "data"})
    evaluator = _build_evaluator(instance, tmp_path)

    with (
        patch("benchmarks.utils.evaluation.create_virtual_key", return_value="sk-test"),
        patch("benchmarks.utils.evaluation.delete_key"),
        patch(
            "benchmarks.utils.evaluation.get_key_spend",
            side_effect=[0.0, 0.125],
        ) as mock_get_key_spend,
        patch("benchmarks.utils.evaluation.time.sleep") as mock_sleep,
    ):
        _, result_output = evaluator._process_one_sync(instance, critic_attempt=1)

    assert result_output.test_result["proxy_cost"] == 0.125
    assert mock_get_key_spend.call_count == 2
    mock_sleep.assert_called_once_with(2)


def test_proxy_cost_retries_after_initial_none_spend(tmp_path):
    instance = EvalInstance(id="test_instance", data={"test": "data"})
    evaluator = _build_evaluator(instance, tmp_path)

    with (
        patch("benchmarks.utils.evaluation.create_virtual_key", return_value="sk-test"),
        patch("benchmarks.utils.evaluation.delete_key"),
        patch(
            "benchmarks.utils.evaluation.get_key_spend",
            side_effect=[None, None, 0.25],
        ) as mock_get_key_spend,
        patch("benchmarks.utils.evaluation.time.sleep") as mock_sleep,
    ):
        _, result_output = evaluator._process_one_sync(instance, critic_attempt=1)

    assert result_output.test_result["proxy_cost"] == 0.25
    assert mock_get_key_spend.call_count == 3
    assert mock_sleep.call_args_list == [call(2), call(4)]


def test_proxy_cost_retry_uses_full_backoff_when_spend_never_appears(tmp_path):
    instance = EvalInstance(id="test_instance", data={"test": "data"})
    evaluator = _build_evaluator(instance, tmp_path)

    with (
        patch("benchmarks.utils.evaluation.create_virtual_key", return_value="sk-test"),
        patch("benchmarks.utils.evaluation.delete_key"),
        patch(
            "benchmarks.utils.evaluation.get_key_spend",
            side_effect=[0.0, 0.0, 0.0, 0.0, 0.0],
        ) as mock_get_key_spend,
        patch("benchmarks.utils.evaluation.time.sleep") as mock_sleep,
    ):
        _, result_output = evaluator._process_one_sync(instance, critic_attempt=1)

    assert result_output.test_result["proxy_cost"] == 0.0
    assert mock_get_key_spend.call_count == 5
    assert mock_sleep.call_args_list == [call(2), call(4), call(8), call(16)]
