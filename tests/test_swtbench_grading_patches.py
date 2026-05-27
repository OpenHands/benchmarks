"""Tests for the local patches applied to upstream swt-bench grading code.

The upstream ``src/grading.py::get_logs_eval`` crashes with ``IndexError`` when
a test log contains the substring ``"trace.py --count -C coverage.cover"`` but
not the full shell-trace marker line. A single bad log then aborts
``make_run_report``, which throws away the whole benchmark report.

We patch the cloned upstream source in
``benchmarks.swtbench.image_utils.ensure_swt_bench_repo`` to:

1. Fall back to ``raw_content`` when the regex split returns a single chunk.
2. Wrap ``report_results`` so any per-instance grading exception becomes an
   "unresolved" marker instead of killing the run.

These tests construct a minimal fake upstream layout, apply the patches, and
verify both behaviors. They are intentionally hermetic — no network, no docker.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.swtbench.image_utils import (
    _GRADING_PATCH_MARKER,
    _REPORT_ISOLATION_MARKER,
    _patch_grading_get_logs_eval,
    _patch_grading_report_results_isolation,
)


_UPSTREAM_GRADING_SNIPPET = """\
import re

APPLY_PATCH_FAIL = "PATCH_APPLY_FAIL"
RESET_FAILED = "RESET_FAILED"
TESTS_ERROR = "TESTS_ERROR"
TESTS_TIMEOUT = "TESTS_TIMEOUT"


def get_logs_eval(log_fp, repo, exec_mode):
    with open(log_fp) as f:
        raw_content = f.read()
    if "trace.py --count -C coverage.cover" in raw_content:
        # NOTE: does not work when not computing coverage
        content = re.split(r"\\n\\+ python3 [^\\n]*trace.py --count -C coverage.cover [^\\n]*\\n", raw_content, flags=re.MULTILINE)[1]
    else:
        content = raw_content
    if "+ cat coverage.cover" in content:
        content = content.split("\\n+ cat coverage.cover")[0]
    if any(x in raw_content for x in [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT]):
        return {}, False
    return {"some-test": "PASSED"}, True


def report_results(
        patch_id,
        run_id,
        golden_code_patch,
        output_paths,
        instance_id,
        repo,
        exec_mode,
):
    # Simulates upstream — calls get_logs_eval; on real bad logs this can raise.
    if output_paths:
        for p in output_paths:
            get_logs_eval(p, repo, exec_mode)
    return {instance_id: {"resolved": True, "coverage_pred": 0.5, "coverage_delta_pred": 0.1, "added_f2p": []}}
"""


def _write_fake_upstream(root: Path) -> Path:
    """Create swt-bench/src/grading.py mimicking the upstream structure."""
    src = root / "src"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    grading = src / "grading.py"
    grading.write_text(_UPSTREAM_GRADING_SNIPPET)
    return root


def _import_grading(swt_bench_dir: Path, monkeypatch):
    """Import the patched grading module from the fake upstream tree."""
    import importlib
    import sys

    monkeypatch.syspath_prepend(str(swt_bench_dir))
    # Make sure we re-import after the source file was rewritten
    for name in [k for k in sys.modules if k.startswith("src")]:
        del sys.modules[name]
    return importlib.import_module("src.grading")


def test_unpatched_get_logs_eval_raises_index_error(tmp_path, monkeypatch):
    """Confirms our reproducer matches the production bug."""
    _write_fake_upstream(tmp_path / "swt-bench")
    g = _import_grading(tmp_path / "swt-bench", monkeypatch)

    bad = tmp_path / "bad.txt"
    # Substring present, but no real "+ python3 ... trace.py ..." marker line.
    bad.write_text("noise trace.py --count -C coverage.cover noise\ntruncated")

    import pytest

    with pytest.raises(IndexError):
        g.get_logs_eval(str(bad), "astropy/astropy", "coverage")


def test_patched_get_logs_eval_falls_back_to_raw_content(tmp_path, monkeypatch):
    swt = _write_fake_upstream(tmp_path / "swt-bench")
    _patch_grading_get_logs_eval(swt)

    grading_text = (swt / "src" / "grading.py").read_text()
    assert _GRADING_PATCH_MARKER in grading_text
    assert "flags=re.MULTILINE)[1]" not in grading_text
    assert "_parts[1] if len(_parts) > 1 else raw_content" in grading_text

    g = _import_grading(swt, monkeypatch)
    bad = tmp_path / "bad.txt"
    bad.write_text("noise trace.py --count -C coverage.cover noise\ntruncated")

    # Must not raise; returns the same shape as the else-branch already does.
    res, applied = g.get_logs_eval(str(bad), "astropy/astropy", "coverage")
    assert isinstance(res, dict)
    assert isinstance(applied, bool)


def test_patches_are_idempotent(tmp_path):
    swt = _write_fake_upstream(tmp_path / "swt-bench")
    _patch_grading_get_logs_eval(swt)
    _patch_grading_report_results_isolation(swt)
    first = (swt / "src" / "grading.py").read_text()

    _patch_grading_get_logs_eval(swt)
    _patch_grading_report_results_isolation(swt)
    second = (swt / "src" / "grading.py").read_text()

    assert first == second
    assert _GRADING_PATCH_MARKER in second
    assert _REPORT_ISOLATION_MARKER in second


def test_patched_report_results_isolates_failures(tmp_path, monkeypatch):
    swt = _write_fake_upstream(tmp_path / "swt-bench")
    _patch_grading_report_results_isolation(swt)

    g = _import_grading(swt, monkeypatch)
    assert hasattr(g, "_openhands_unsafe_report_results")

    def _boom(*_a, **_k):
        raise RuntimeError("simulated grading bug")

    monkeypatch.setattr(g, "_openhands_unsafe_report_results", _boom)

    out = g.report_results(
        "patch_id",
        "run_id",
        "",
        None,
        "fake__instance-1",
        "astropy/astropy",
        "coverage",
    )

    assert out == {
        "fake__instance-1": {
            "resolved": False,
            "coverage_pred": None,
            "coverage_delta_pred": 0,
            "added_f2p": [],
        }
    }


def test_patch_helpers_no_op_when_grading_missing(tmp_path):
    """Patch helpers must not raise if the upstream source is unexpectedly absent."""
    empty = tmp_path / "swt-bench"
    empty.mkdir()
    # Should log a warning but not raise.
    _patch_grading_get_logs_eval(empty)
    _patch_grading_report_results_isolation(empty)
