"""Tests for patch_utils.keep_only_test_files."""

from benchmarks.utils.patch_utils import keep_only_test_files


def _diff(path: str, body: str = "@@ -1 +1 @@\n-old\n+new\n") -> str:
    """Build a minimal ``diff --git`` block for ``path``."""
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n{body}"


class TestKeepOnlyTestFiles:
    def test_empty_returns_empty(self):
        assert keep_only_test_files("") == ""

    def test_passthrough_when_no_diff_header(self):
        patch = "not a real patch\n"
        assert keep_only_test_files(patch) == patch

    def test_keeps_files_under_tests_dir(self):
        patch = _diff("tests/foo/test_bar.py") + _diff("sympy/core/basic.py")
        out = keep_only_test_files(patch)
        assert "tests/foo/test_bar.py" in out
        assert "sympy/core/basic.py" not in out

    def test_keeps_files_under_testing_dir(self):
        patch = _diff("django/testing/helpers.py")
        assert "django/testing/helpers.py" in keep_only_test_files(patch)

    def test_keeps_files_under_singular_test_dir(self):
        # ``test/`` (singular) at a non-root level is a valid test layout
        # too (used by e.g. requests, pytest itself); document that it's kept.
        patch = _diff("test/unit/helpers.py")
        assert "test/unit/helpers.py" in keep_only_test_files(patch)

    def test_keeps_conftest(self):
        patch = _diff("pkg/conftest.py")
        assert "pkg/conftest.py" in keep_only_test_files(patch)

    def test_keeps_test_prefix_outside_test_dir(self):
        patch = _diff("pkg/sub/test_helpers.py")
        assert "pkg/sub/test_helpers.py" in keep_only_test_files(patch)

    def test_keeps_underscore_test_suffix(self):
        patch = _diff("pkg/sub/helpers_test.py")
        assert "pkg/sub/helpers_test.py" in keep_only_test_files(patch)

    def test_drops_root_level_scratch_repros(self):
        # Agent scratch files at the repo root are not real tests even when
        # they match the naming convention.
        patch = (
            _diff("reproduction.py")
            + _diff("test_repro.py")
            + _diff("FIX_SUMMARY.md")
            + _diff("tests/test_real.py")
        )
        out = keep_only_test_files(patch)
        assert "reproduction.py" not in out
        assert "test_repro.py" not in out
        assert "FIX_SUMMARY.md" not in out
        assert "tests/test_real.py" in out

    def test_drops_root_level_conftest(self):
        # A root-level ``conftest.py`` is treated as agent scratch (the SWT
        # repos we evaluate keep conftest under a package), not as a real
        # test file. This test pins that trade-off so a future loosening of
        # the root-exclusion guard is intentional, not accidental.
        patch = _diff("conftest.py") + _diff("tests/conftest.py")
        out = keep_only_test_files(patch)
        assert "diff --git a/conftest.py b/conftest.py" not in out
        assert "tests/conftest.py" in out

    def test_drops_build_and_docs(self):
        patch = (
            _diff("build/lib/requests/__init__.py")
            + _diff("docs/intro.rst")
            + _diff("tests/test_real.py")
        )
        out = keep_only_test_files(patch)
        assert "build/lib/requests/__init__.py" not in out
        assert "docs/intro.rst" not in out
        assert "tests/test_real.py" in out

    def test_all_non_test_returns_empty_string(self):
        patch = _diff("sympy/core/basic.py") + _diff("django/db/models/base.py")
        assert keep_only_test_files(patch) == ""
