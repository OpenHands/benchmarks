"""Tests for parse_report_summary in benchmarks/commit0/run_infer.py.

Validates parsing of pytest-json-report summary format, including edge cases.
All test data matches the actual output of pytest-json-report v1.5.0.
Keys like "failed", "skipped", "xfailed", "xpassed" only appear when non-zero.
"""

import json
import subprocess

import pytest

from benchmarks.commit0.run_infer import EXTRACT_SUMMARY_SCRIPT, parse_report_summary


@pytest.mark.parametrize(
    "summary, expected_passed, expected_tests, expected_ratio, expected_duration",
    [
        (
            {"passed": 6704, "total": 6704, "collected": 6704, "duration": 120.5},
            6704,
            6704,
            1.0,
            120.5,
        ),
        (
            {
                "passed": 100,
                "failed": 5,
                "skipped": 3,
                "total": 108,
                "collected": 108,
                "duration": 45.2,
            },
            100,
            108,
            100 / 108,
            45.2,
        ),
        (
            {"failed": 10, "total": 10, "collected": 10, "duration": 2.0},
            0,
            10,
            0.0,
            2.0,
        ),
        (
            {"skipped": 5, "total": 5, "collected": 5, "duration": 0.1},
            0,
            5,
            0.0,
            0.1,
        ),
    ],
    ids=["all_passed", "mixed_results", "all_failed", "only_skipped"],
)
def test_basic_summaries(
    summary, expected_passed, expected_tests, expected_ratio, expected_duration
):
    result = parse_report_summary(json.dumps(summary))
    assert result["num_passed"] == expected_passed
    assert result["num_tests"] == expected_tests
    assert result["passed"] == pytest.approx(expected_ratio)
    assert result["sum"] == expected_duration


@pytest.mark.parametrize(
    "summary, expected_passed, expected_tests",
    [
        (
            {"passed": 10, "xfailed": 3, "total": 13, "collected": 13, "duration": 5.0},
            13,
            13,
        ),
        (
            {"passed": 5, "xpassed": 2, "total": 7, "collected": 7, "duration": 1.0},
            5,
            7,
        ),
        (
            {"xfailed": 4, "total": 4, "collected": 4, "duration": 1.5},
            4,
            4,
        ),
    ],
    ids=["xfailed_counts_as_passed", "xpassed_not_counted", "xfailed_only"],
)
def test_xfail_handling(summary, expected_passed, expected_tests):
    result = parse_report_summary(json.dumps(summary))
    assert result["num_passed"] == expected_passed
    assert result["num_tests"] == expected_tests


@pytest.mark.parametrize(
    "raw_json, error_type",
    [
        ("{}", ValueError),
        (json.dumps({"total": 0, "collected": 0}), ValueError),
        (json.dumps({"passed": 5, "collected": 5}), ValueError),
        ("not json at all", json.JSONDecodeError),
        ('{"passed": 5, "total":', json.JSONDecodeError),
    ],
    ids=[
        "empty_summary",
        "zero_total",
        "missing_total",
        "invalid_json",
        "truncated_json",
    ],
)
def test_validation_raises(raw_json, error_type):
    with pytest.raises(error_type):
        parse_report_summary(raw_json)


@pytest.mark.parametrize(
    "summary, expected_duration",
    [
        ({"passed": 1, "total": 1, "collected": 1, "duration": 99.9}, 99.9),
        ({"passed": 1, "total": 1, "collected": 1}, 0),
    ],
    ids=["duration_present", "duration_missing_defaults_to_zero"],
)
def test_duration(summary, expected_duration):
    result = parse_report_summary(json.dumps(summary))
    assert result["sum"] == expected_duration


@pytest.mark.parametrize(
    "summary, expected_passed",
    [
        ({"failed": 3, "total": 3, "collected": 3, "duration": 1.0}, 0),
        ({"passed": 10, "total": 10, "collected": 10, "duration": 2.0}, 10),
    ],
    ids=["no_passed_key", "no_xfailed_key"],
)
def test_missing_optional_keys(summary, expected_passed):
    result = parse_report_summary(json.dumps(summary))
    assert result["num_passed"] == expected_passed


def test_real_pytest_json_report_output():
    """End-to-end test using actual pytest-json-report v1.5.0 output.

    This simulates what the in-container command produces: the summary dict
    from a real report.json with 'duration' injected from the top level.
    """
    # Actual output from: python3 -m pytest --json-report test_sample.py
    # (1 passed, 1 failed, 1 skipped), then extracted via the in-container command.
    raw = (
        '{"passed": 1, "failed": 1, "skipped": 1, '
        '"total": 3, "collected": 3, "duration": 0.08435988426208496}'
    )
    result = parse_report_summary(raw)
    assert result["num_passed"] == 1
    assert result["num_tests"] == 3
    assert result["passed"] == pytest.approx(1 / 3)
    assert result["sum"] == pytest.approx(0.0844, abs=1e-3)


# ---------------------------------------------------------------------------
# Integration tests: run the actual extraction one-liner against report.json
# files on disk, then feed the output into parse_report_summary.
# ---------------------------------------------------------------------------


def _run_extraction(report_dir: str) -> subprocess.CompletedProcess:
    """Run the extraction one-liner against a report.json in *report_dir*."""
    return subprocess.run(
        ["python3", "-c", EXTRACT_SUMMARY_SCRIPT],
        cwd=report_dir,
        capture_output=True,
        text=True,
    )


def test_extraction_full_report(tmp_path):
    """One-liner extracts summary + duration from a full pytest-json-report."""
    report = {
        "created": 1700000000.0,
        "duration": 120.5,
        "exitcode": 0,
        "summary": {"passed": 100, "failed": 2, "total": 102, "collected": 102},
        "tests": [{"nodeid": "test_a.py::test_x", "outcome": "passed"}],
    }
    (tmp_path / "report.json").write_text(json.dumps(report))

    proc = _run_extraction(str(tmp_path))
    assert proc.returncode == 0

    result = parse_report_summary(proc.stdout)
    assert result["num_passed"] == 100
    assert result["num_tests"] == 102
    assert result["sum"] == 120.5


def test_extraction_no_summary_field_fails_loudly(tmp_path):
    """When report.json has no summary, the flow raises instead of scoring 0/0."""
    report = {"tests": [], "duration": 1.0}
    (tmp_path / "report.json").write_text(json.dumps(report))

    proc = _run_extraction(str(tmp_path))
    # One-liner succeeds (it handles missing summary with .get()), but
    # parse_report_summary must reject the empty result.
    assert proc.returncode == 0

    with pytest.raises(ValueError, match="missing or empty 'total'"):
        parse_report_summary(proc.stdout)


def test_extraction_missing_report_file_fails(tmp_path):
    """When report.json doesn't exist, the one-liner fails with non-zero exit."""
    proc = _run_extraction(str(tmp_path))
    assert proc.returncode != 0


def test_extraction_corrupt_report_file_fails(tmp_path):
    """When report.json is not valid JSON, the one-liner fails with non-zero exit."""
    (tmp_path / "report.json").write_text("this is not json {{{")

    proc = _run_extraction(str(tmp_path))
    assert proc.returncode != 0
