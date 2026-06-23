"""Tests for the generic Harbor run_infer helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.harbor.run_infer import (
    _is_sensitive_value,
    _load_task_ids,
    _parse_key_value,
    _resolve_target,
    _split_json_values,
    _target_args,
)


def test_load_task_ids_strips_and_ignores_comments(tmp_path: Path) -> None:
    f = tmp_path / "tasks.txt"
    f.write_text("# comment\n  task_a  \n\ntask_b\n", encoding="utf-8")
    assert _load_task_ids(str(f)) == ["task_a", "task_b"]


def test_target_args_dataset() -> None:
    assert _target_args("foo", "dataset") == ["-d", "foo"]


def test_target_args_config() -> None:
    assert _target_args("foo.yaml", "config") == ["-c", "foo.yaml"]


def test_target_args_path() -> None:
    assert _target_args("some/path", "path") == ["-p", "some/path"]


def test_target_args_invalid() -> None:
    with pytest.raises(ValueError, match="Unsupported Harbor target type"):
        _target_args("foo", "bogus")


def test_parse_key_value_ok() -> None:
    assert _parse_key_value(["A=1", "B=2"]) == ["A=1", "B=2"]


def test_parse_key_value_missing_equals() -> None:
    with pytest.raises(ValueError, match="Expected KEY=VALUE"):
        _parse_key_value(["bad"])


def test_split_json_values_none() -> None:
    assert _split_json_values(None) == []


def test_split_json_values_dict() -> None:
    assert _split_json_values('{"A": "1", "B": "2"}') == ["A=1", "B=2"]


def test_split_json_values_list() -> None:
    assert _split_json_values('["A=1", "B=2"]') == ["A=1", "B=2"]


def test_split_json_values_invalid() -> None:
    with pytest.raises(ValueError, match="Expected a JSON object or list"):
        _split_json_values('"not-an-object-or-list"')


def _make_args(**kwargs: object) -> argparse.Namespace:
    defaults: dict[str, object] = dict(
        harbor_target=None,
        harbor_target_type="auto",
        harbor_adapter_repo=None,
        harbor_adapter_ref=None,
        harbor_adapter_path=None,
    )
    defaults.update(kwargs.items())
    return argparse.Namespace(**defaults)


def test_resolve_target_requires_target() -> None:
    with pytest.raises(
        RuntimeError, match="A Harbor target or adapter path is required"
    ):
        _resolve_target(_make_args())


def test_resolve_target_dataset_auto() -> None:
    # When target doesn't exist on disk and type is auto, defaults to dataset
    target, target_type, checkout, sha = _resolve_target(
        _make_args(harbor_target="my-dataset")
    )
    assert target == "my-dataset"
    assert target_type == "dataset"
    assert checkout is None
    assert sha is None


def test_resolve_target_config_auto(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("foo: bar", encoding="utf-8")
    target, target_type, checkout, sha = _resolve_target(
        _make_args(harbor_target=str(cfg))
    )
    assert target == str(cfg)
    assert target_type == "config"
    assert checkout is None
    assert sha is None


def test_resolve_target_path_auto(tmp_path: Path) -> None:
    p = tmp_path / "some_dir"
    p.mkdir()
    target, target_type, checkout, sha = _resolve_target(
        _make_args(harbor_target=str(p))
    )
    assert target == str(p)
    assert target_type == "path"
    assert checkout is None
    assert sha is None


def test_resolve_target_explicit_type() -> None:
    target, target_type, checkout, sha = _resolve_target(
        _make_args(harbor_target="ds", harbor_target_type="dataset")
    )
    assert target == "ds"
    assert target_type == "dataset"
    assert checkout is None
    assert sha is None


def test_resolve_target_adapter_path(tmp_path: Path) -> None:
    """Adapter path resolution clones the repo and resolves the target inside it."""
    fake_checkout = tmp_path / "fake-clone"
    fake_checkout.mkdir()
    cfg_inside = fake_checkout / "adapter.yaml"
    cfg_inside.write_text("foo: bar", encoding="utf-8")

    with patch(
        "benchmarks.harbor.run_infer._checkout_adapter",
        return_value=(fake_checkout, "abc123def456"),
    ):
        target, target_type, checkout, sha = _resolve_target(
            _make_args(
                harbor_adapter_repo="https://example.com/repo.git",
                harbor_adapter_path="adapter.yaml",
            )
        )

    assert target == str(cfg_inside)
    assert target_type == "config"
    assert checkout == str(fake_checkout)
    assert sha == "abc123def456"


# --- Secret masking tests ---


def test_is_sensitive_value_key_suffix() -> None:
    assert _is_sensitive_value("--ae", "LLM_API_KEY=secret123")


def test_is_sensitive_value_token() -> None:
    assert _is_sensitive_value("--ae", "GITHUB_TOKEN=ghp_xxx")


def test_is_sensitive_value_secret() -> None:
    assert _is_sensitive_value("--ae", "HF_SECRET=hf_xxx")


def test_is_sensitive_value_password() -> None:
    assert _is_sensitive_value("--ae", "DB_PASSWORD=p455w0rd")


def test_is_sensitive_value_ak_token() -> None:
    """--ak values with secret-like keys must also be masked (regression test)."""
    assert _is_sensitive_value("--ak", "GITHUB_TOKEN=ghp_xxx")


def test_is_sensitive_value_ak_key_suffix() -> None:
    assert _is_sensitive_value("--ak", "API_KEY=secret123")


def test_is_sensitive_value_ak_non_secret() -> None:
    assert not _is_sensitive_value("--ak", "MAX_ITERATIONS=100")


def test_is_sensitive_value_unknown_flag() -> None:
    assert not _is_sensitive_value("--foo", "GITHUB_TOKEN=ghp_xxx")


def test_is_sensitive_value_non_secret() -> None:
    assert not _is_sensitive_value("--ae", "LLM_BASE_URL=https://api.example.com")


def test_is_sensitive_value_non_secret_env() -> None:
    assert not _is_sensitive_value("--ae", "MAX_ITERATIONS=100")
