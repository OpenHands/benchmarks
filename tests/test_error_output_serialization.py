"""Tests for _create_error_output numpy serialization fix.

Regression test for: error outputs containing pandas row dicts (which hold
numpy scalars / arrays) silently failed to serialize, causing all failed
instances to be dropped and never recorded in the attempt file.
"""

import json

import numpy as np

from benchmarks.utils.evaluation import _to_serializable


# ---------------------------------------------------------------------------
# _to_serializable unit tests
# ---------------------------------------------------------------------------


def test_to_serializable_numpy_integer():
    assert _to_serializable(np.int64(42)) == 42
    assert isinstance(_to_serializable(np.int64(42)), int)


def test_to_serializable_numpy_floating():
    result = _to_serializable(np.float64(3.14))
    assert abs(result - 3.14) < 1e-10
    assert isinstance(result, float)


def test_to_serializable_numpy_bool():
    assert _to_serializable(np.bool_(True)) is True
    assert isinstance(_to_serializable(np.bool_(True)), bool)


def test_to_serializable_numpy_ndarray():
    arr = np.array([1, 2, 3])
    result = _to_serializable(arr)
    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_to_serializable_nested_dict():
    data = {
        "id": np.int64(7),
        "score": np.float32(0.95),
        "tags": np.array(["a", "b"]),
        "nested": {"value": np.int32(1)},
    }
    result = _to_serializable(data)
    # Must round-trip through json without error
    serialized = json.dumps(result)
    parsed = json.loads(serialized)
    assert parsed["id"] == 7
    assert abs(parsed["score"] - 0.95) < 1e-3
    assert parsed["tags"] == ["a", "b"]
    assert parsed["nested"]["value"] == 1


def test_to_serializable_plain_python_passthrough():
    data = {"a": 1, "b": "hello", "c": [True, None, 3.14]}
    assert _to_serializable(data) == data


def test_to_serializable_list_and_tuple():
    lst = [np.int64(1), np.float64(2.0)]
    assert _to_serializable(lst) == [1, 2.0]

    tup = (np.int64(1), np.float64(2.0))
    result = _to_serializable(tup)
    assert result == (1, 2.0)
    assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# Integration: _create_error_output produces JSON-serializable output
# ---------------------------------------------------------------------------


def _make_eval(tmp_path):
    """Instantiate a minimal concrete Evaluation subclass for testing."""
    from unittest.mock import MagicMock

    from benchmarks.utils.evaluation import Evaluation
    from benchmarks.utils.models import EvalMetadata

    class _DummyEval(Evaluation):
        def prepare_instances(self):
            return []

        def prepare_workspace(self, instance, resource_factor=1, forward_env=None):
            raise NotImplementedError

        def evaluate_instance(self, instance, workspace):
            raise NotImplementedError

    llm_mock = MagicMock()
    llm_mock.model = "test-model"
    llm_mock.model_dump_json.return_value = "{}"
    llm_mock.model_copy.return_value = llm_mock
    llm_mock.temperature = 0.0

    critic_mock = MagicMock()
    critic_mock.model_dump.return_value = {"kind": "mock"}

    metadata = EvalMetadata(
        llm=llm_mock,
        dataset="test_ds",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir=str(tmp_path),
        details={},
        eval_limit=1,
        max_retries=1,
        critic=critic_mock,
    )
    return _DummyEval(metadata=metadata, num_workers=1)


def test_create_error_output_serializes_numpy_instance_data(tmp_path):
    """_create_error_output must produce an EvalOutput that serializes cleanly
    even when instance.data contains numpy types (as it does after
    pandas row.to_dict())."""
    from benchmarks.utils.models import EvalInstance

    eval_ = _make_eval(tmp_path)

    # Simulate a pandas row.to_dict() result with numpy types
    numpy_data = {
        "instance_id": "django__django-1234",
        "repo": "django/django",
        "int_col": np.int64(99),
        "float_col": np.float64(1.5),
        "bool_col": np.bool_(False),
        "array_col": np.array([10, 20, 30]),
    }
    instance = EvalInstance(id="django__django-1234", data=numpy_data)

    out = eval_._create_error_output(instance, RuntimeError("boom"), retry_count=3)

    # Must not raise
    serialized = out.model_dump_json()
    parsed = json.loads(serialized)

    assert parsed["instance_id"] == "django__django-1234"
    assert parsed["error"] is not None
    inst = parsed["instance"]
    assert inst["int_col"] == 99
    assert isinstance(inst["int_col"], int)
    assert inst["array_col"] == [10, 20, 30]
