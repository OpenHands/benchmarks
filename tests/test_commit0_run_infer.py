"""Tests for commit0 run_infer test command helpers."""

import pytest

from benchmarks.commit0.run_infer import get_pythonpath_prefix, normalize_pytest_cmd


@pytest.mark.parametrize(
    "input_cmd, expected",
    [
        ("pytest", "python -m pytest"),
        ("pytest3", "python -m pytest3"),
        ("python -m pytest", "python -m pytest"),
        ("mypytest", "mypytest"),
        ("pytest-xdist", "pytest-xdist"),
        ("pytest_runner", "pytest_runner"),
        (
            "pytest --assert=plain --ignore=setup.py",
            "python -m pytest --assert=plain --ignore=setup.py",
        ),
    ],
    ids=[
        "bare_pytest",
        "bare_pytest3",
        "already_module_form",
        "substring_mypytest",
        "substring_pytest-xdist",
        "substring_pytest_runner",
        "real-parsel-scenario",
    ],
)
def test_normalize_pytest_cmd(input_cmd, expected):
    assert normalize_pytest_cmd(input_cmd) == expected


@pytest.mark.parametrize(
    "src_dir, expected",
    [
        ("src/cachetools", "PYTHONPATH=src:$PYTHONPATH "),
        ("src", "PYTHONPATH=src:$PYTHONPATH "),
        ("", ""),
        ("lib/mypackage", ""),
        ("tests/src/data", ""),
    ],
    ids=[
        "src_layout",
        "bare_src",
        "empty_string",
        "no_src_dir",
        "src_not_at_start",
    ],
)
def test_get_pythonpath_prefix(src_dir, expected):
    assert get_pythonpath_prefix(src_dir) == expected
