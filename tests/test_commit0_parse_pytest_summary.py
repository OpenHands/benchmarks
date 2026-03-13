"""Tests for the _parse_pytest_summary helper in benchmarks/commit0/run_infer.py."""

import pytest

from benchmarks.commit0.run_infer import _parse_pytest_summary


@pytest.mark.parametrize(
    "output, expected",
    [
        # Basic passing
        ("===== 6704 passed in 120.34s =====", (6704, 6704)),
        ("===== 1 passed in 0.50s =====", (1, 1)),
        # Mixed results
        ("===== 6704 passed, 5 failed, 2 skipped in 120.34s =====", (6704, 6711)),
        ("===== 10 passed, 3 failed in 1.23s =====", (10, 13)),
        ("===== 50 passed, 10 skipped in 5.0s =====", (50, 60)),
        ("===== 99999 passed, 1 failed, 100 skipped in 999.99s =====", (99999, 100100)),
        # Only failures / skipped
        ("===== 5 failed in 1.23s =====", (0, 5)),
        ("===== 7 skipped in 0.05s =====", (0, 7)),
        ("===== 3 failed, 2 skipped in 0.42s =====", (0, 5)),
    ],
    ids=[
        "single_passed",
        "one_passed",
        "passed_failed_skipped",
        "passed_and_failed",
        "passed_and_skipped",
        "large_counts",
        "only_failed",
        "only_skipped",
        "failed_and_skipped",
    ],
)
def test_parse_summary(output, expected):
    assert _parse_pytest_summary(output) == expected


@pytest.mark.parametrize(
    "output, expected",
    [
        ("===== 10 passed, 3 xfail in 5.0s =====", (13, 13)),
        ("===== 4 xfail in 2.0s =====", (4, 4)),
        ("===== 8 passed, 2 failed, 1 xfail in 3.1s =====", (9, 11)),
        ("===== 100 passed, 10 failed, 5 skipped, 3 xfail in 60.0s =====", (103, 118)),
        # xpass counts toward total but NOT passed
        ("===== 5 passed, 2 xpass in 1.0s =====", (5, 7)),
    ],
    ids=[
        "passed_and_xfail",
        "xfail_only",
        "passed_failed_xfail",
        "all_categories",
        "xpass_not_counted_as_passed",
    ],
)
def test_xfail_counting(output, expected):
    assert _parse_pytest_summary(output) == expected


@pytest.mark.parametrize(
    "output",
    [
        "",
        "   \n\t  ",
        "some random test output without pytest summary",
        "====== some text ======",
        # 0 passed → total == 0 → returns None
        "===== 0 passed in 0.01s =====",
    ],
    ids=[
        "empty_string",
        "whitespace_only",
        "no_delimiter",
        "partial_delimiter_no_timing",
        "zero_total",
    ],
)
def test_returns_none(output):
    assert _parse_pytest_summary(output) is None


@pytest.mark.parametrize(
    "output",
    [
        "= 2 passed in 0.1s =",
        "======= 3 passed in 0.2s =======",
        f"{'=' * 40} 5 passed in 1.0s {'=' * 40}",
    ],
    ids=["short", "medium", "wide_80"],
)
def test_separator_variants(output):
    assert _parse_pytest_summary(output) is not None


@pytest.mark.parametrize(
    "output",
    [
        "===== 1 passed in 1s =====",
        "===== 1 passed in 1.5s =====",
        "===== 1 passed in 1.123456s =====",
    ],
    ids=["integer", "one_decimal", "many_decimals"],
)
def test_timing_formats(output):
    assert _parse_pytest_summary(output) == (1, 1)


@pytest.mark.parametrize(
    "output, expected",
    [
        pytest.param(
            "collecting ... collected 6711 items\n\n"
            "tests/test_foo.py::test_alpha PASSED\n"
            "tests/test_foo.py::test_beta FAILED\n"
            "tests/test_bar.py::test_gamma SKIPPED\n\n"
            "======= short test summary info ========\n"
            "FAILED tests/test_foo.py::test_beta - AssertionError\n"
            "======= 6704 passed, 5 failed, 2 skipped in 120.34s ========\n",
            (6704, 6711),
            id="full_pytest_output",
        ),
        pytest.param(
            "some header text\n"
            "FAILED tests/module.py::test_x\n"
            "====== 3 passed, 1 failed in 0.5s ======\n"
            "some trailing text\n",
            (3, 4),
            id="summary_mid_output",
        ),
        pytest.param(
            "ERROR collecting tests/bad_module.py\n"
            "ImportError: cannot import name 'foo'\n"
            "====== 1 error in 0.12s ======\n",
            (0, 1),
            id="collection_errors",
        ),
        pytest.param(
            "====== 1 passed in 0.1s ======\n====== 2 passed in 0.2s ======\n",
            (1, 1),
            id="multiple_summary_lines_picks_first",
        ),
        pytest.param(
            "PASSED test_x  ===== 4 passed, 1 failed in 2.0s =====",
            (4, 5),
            id="inline_no_newlines",
        ),
    ],
)
def test_realistic_output(output, expected):
    assert _parse_pytest_summary(output) == expected
