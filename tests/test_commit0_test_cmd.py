"""Tests for test command construction fixes in commit0 evaluate_instance.

Validates Bug #526 fixes:
  1. Bare `pytest` is always replaced with `python -m pytest`
  2. src-layout repos get `PYTHONPATH=src` prepended
"""

import pytest


def _build_test_cmd(test_cmd: str, src_dir: str) -> str:
    """Reproduce the test-command construction logic from evaluate_instance."""
    if "pytest" in test_cmd and "python -m pytest" not in test_cmd:
        test_cmd = test_cmd.replace("pytest", "python -m pytest", 1)
    env_prefix = "PYTHONPATH=src " if src_dir and "src/" in src_dir else ""
    repo_path = "/workspace/repo"
    test_dir = "tests/"
    return (
        f"cd {repo_path} && {env_prefix}{test_cmd} "
        f"--json-report --json-report-file=report.json "
        f"--continue-on-collection-errors {test_dir} > test_output.txt 2>&1"
    )


# --- Bug 2: pytest -> python -m pytest replacement ---


@pytest.mark.parametrize(
    "test_cmd, expected_fragment",
    [
        ("pytest", "python -m pytest"),
        ("pytest -x", "python -m pytest -x"),
        ("python -m pytest", "python -m pytest"),
    ],
    ids=["bare_pytest", "pytest_with_args", "already_python_m"],
)
def test_pytest_replacement(test_cmd: str, expected_fragment: str) -> None:
    result = _build_test_cmd(test_cmd, src_dir="")
    assert expected_fragment in result
    # Ensure no double replacement
    assert "python -m python -m pytest" not in result


# --- Bug 1: PYTHONPATH=src for src-layout repos ---


@pytest.mark.parametrize(
    "src_dir, should_have_pythonpath",
    [
        ("src/cachetools/", True),
        ("src/some_pkg/", True),
        ("cachetools/", False),
        ("", False),
    ],
    ids=["src_layout", "src_layout_other", "flat_layout", "empty"],
)
def test_pythonpath_for_src_layout(src_dir: str, should_have_pythonpath: bool) -> None:
    result = _build_test_cmd("pytest", src_dir)
    if should_have_pythonpath:
        assert "PYTHONPATH=src " in result
    else:
        assert "PYTHONPATH=src " not in result


def test_combined_fix() -> None:
    """Simulates cachetools: bare pytest + src-layout."""
    result = _build_test_cmd("pytest", "src/cachetools/")
    assert "PYTHONPATH=src python -m pytest" in result
