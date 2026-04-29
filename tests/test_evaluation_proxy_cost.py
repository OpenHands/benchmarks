"""Tests for proxy cost retrieval in the evaluation worker."""

import base64
import json
import tarfile
from io import BytesIO
from types import SimpleNamespace
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


def _conversation_archive_base64(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int,
    accumulated_cost: float,
) -> str:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as conv_tar:
        base_state = {
            "stats": {
                "usage_to_metrics": {
                    "default": {
                        "model_name": "test-model",
                        "accumulated_cost": accumulated_cost,
                        "accumulated_token_usage": {
                            "model": "test-model",
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "cache_read_tokens": cache_read_tokens,
                            "cache_write_tokens": 0,
                            "reasoning_tokens": 0,
                            "context_window": 0,
                            "per_turn_token": prompt_tokens + completion_tokens,
                            "response_id": "",
                        },
                    }
                }
            }
        }
        payload = json.dumps(base_state).encode("utf-8")
        info = tarfile.TarInfo("workspace/conversations/test/base_state.json")
        info.size = len(payload)
        conv_tar.addfile(info, BytesIO(payload))
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_evaluator(
    instance: EvalInstance,
    tmp_path,
    *,
    evaluate_side_effect: Exception | None = None,
    conversation_archive_base64: str | None = None,
):
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
            workspace.execute_command = Mock(
                return_value=SimpleNamespace(
                    exit_code=0,
                    stdout=conversation_archive_base64 or "",
                )
            )
            return workspace

        def evaluate_instance(self, instance, workspace):
            if evaluate_side_effect is not None:
                raise evaluate_side_effect
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


def test_final_failure_persists_proxy_cost_and_recovered_metrics(tmp_path):
    instance = EvalInstance(id="test_instance", data={"test": "data"})
    evaluator = _build_evaluator(
        instance,
        tmp_path,
        evaluate_side_effect=RuntimeError("boom"),
        conversation_archive_base64=_conversation_archive_base64(
            prompt_tokens=100,
            completion_tokens=25,
            cache_read_tokens=10,
            accumulated_cost=1.75,
        ),
    )

    with (
        patch("benchmarks.utils.evaluation.create_virtual_key", return_value="sk-test"),
        patch("benchmarks.utils.evaluation.delete_key"),
        patch(
            "benchmarks.utils.evaluation.get_key_spend",
            side_effect=[0.0, 0.5],
        ) as mock_get_key_spend,
        patch("benchmarks.utils.evaluation.time.sleep") as mock_sleep,
    ):
        _, result_output = evaluator._process_one_sync(instance, critic_attempt=1)

    assert result_output.error is not None
    assert result_output.test_result["proxy_cost"] == 0.5
    assert result_output.metrics is not None
    assert result_output.metrics.accumulated_cost == 1.75
    assert result_output.metrics.accumulated_token_usage is not None
    assert result_output.metrics.accumulated_token_usage.prompt_tokens == 100
    assert result_output.metrics.accumulated_token_usage.completion_tokens == 25
    assert result_output.metrics.accumulated_token_usage.cache_read_tokens == 10
    assert mock_get_key_spend.call_count == 2
    mock_sleep.assert_called_once_with(2)


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
