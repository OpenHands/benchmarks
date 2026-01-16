"""Tests for report_costs.py functionality.

This test suite verifies that:
1. The calculate_time_statistics function returns total_duration
2. The summary in calculate_costs uses total_duration instead of average_duration
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.utils.report_costs import (
    calculate_line_duration,
    calculate_time_statistics,
    extract_accumulated_cost,
)


def create_test_entry(
    instance_id: str,
    cost: float = 0.0,
    duration_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Create a test JSONL entry with specified cost and duration."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    end_time = base_time + timedelta(seconds=duration_seconds)

    return {
        "instance_id": instance_id,
        "metrics": {"accumulated_cost": cost},
        "history": [
            {"timestamp": base_time.isoformat()},
            {"timestamp": end_time.isoformat()},
        ],
    }


class TestCalculateTimeStatistics:
    """Tests for calculate_time_statistics function."""

    def test_returns_total_duration(self) -> None:
        """Test that calculate_time_statistics returns total_duration."""
        entries: List[Optional[Dict[str, Any]]] = [
            create_test_entry("test-1", cost=1.0, duration_seconds=60),
            create_test_entry("test-2", cost=2.0, duration_seconds=120),
            create_test_entry("test-3", cost=3.0, duration_seconds=180),
        ]

        stats = calculate_time_statistics(entries)

        assert "total_duration" in stats, "Should return total_duration"
        assert stats["total_duration"] == 360.0, (
            "Total duration should be sum of all durations (60+120+180=360)"
        )

    def test_total_duration_with_empty_data(self) -> None:
        """Test that total_duration is 0.0 for empty data."""
        stats = calculate_time_statistics([])

        assert "total_duration" in stats
        assert stats["total_duration"] == 0.0

    def test_total_duration_with_no_valid_durations(self) -> None:
        """Test that total_duration is 0.0 when no entries have valid durations."""
        entries: List[Optional[Dict[str, Any]]] = [
            {"instance_id": "test-1", "metrics": {"accumulated_cost": 1.0}},
            {"instance_id": "test-2", "metrics": {"accumulated_cost": 2.0}},
        ]

        stats = calculate_time_statistics(entries)

        assert "total_duration" in stats
        assert stats["total_duration"] == 0.0

    def test_average_duration_still_calculated(self) -> None:
        """Test that average_duration is still calculated for backward compatibility."""
        entries: List[Optional[Dict[str, Any]]] = [
            create_test_entry("test-1", cost=1.0, duration_seconds=60),
            create_test_entry("test-2", cost=2.0, duration_seconds=120),
        ]

        stats = calculate_time_statistics(entries)

        assert "average_duration" in stats
        assert stats["average_duration"] == 90.0, "Average should be (60+120)/2=90"
        assert stats["total_duration"] == 180.0, "Total should be 60+120=180"


class TestCalculateLineDuration:
    """Tests for calculate_line_duration function."""

    def test_calculates_duration_correctly(self) -> None:
        """Test that duration is calculated from timestamps."""
        entry = create_test_entry("test-1", duration_seconds=300)

        duration = calculate_line_duration(entry)

        assert duration == 300.0

    def test_returns_none_for_missing_history(self) -> None:
        """Test that None is returned when history is missing."""
        entry: Dict[str, Any] = {"instance_id": "test-1"}

        duration = calculate_line_duration(entry)

        assert duration is None

    def test_returns_none_for_none_entry(self) -> None:
        """Test that None is returned for None entry."""
        duration = calculate_line_duration(None)

        assert duration is None


class TestExtractAccumulatedCost:
    """Tests for extract_accumulated_cost function."""

    def test_sums_costs_correctly(self) -> None:
        """Test that costs are summed correctly."""
        entries: List[Optional[Dict[str, Any]]] = [
            {"metrics": {"accumulated_cost": 1.5}},
            {"metrics": {"accumulated_cost": 2.5}},
            {"metrics": {"accumulated_cost": 3.0}},
        ]

        total = extract_accumulated_cost(entries)

        assert total == 7.0

    def test_handles_empty_data(self) -> None:
        """Test that empty data returns 0.0."""
        total = extract_accumulated_cost([])

        assert total == 0.0


class TestSummaryTotalDuration:
    """Integration tests for summary total_duration in calculate_costs."""

    def test_summary_contains_total_duration(self) -> None:
        """Test that the summary in cost_report.jsonl contains total_duration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test output.jsonl file
            output_file = tmpdir_path / "output.jsonl"
            entries = [
                create_test_entry("test-1", cost=1.0, duration_seconds=60),
                create_test_entry("test-2", cost=2.0, duration_seconds=120),
            ]
            with open(output_file, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

            # Import and run calculate_costs
            from benchmarks.utils.report_costs import calculate_costs

            calculate_costs(str(tmpdir_path))

            # Read the generated cost_report.jsonl
            report_file = tmpdir_path / "cost_report.jsonl"
            assert report_file.exists(), "cost_report.jsonl should be created"

            with open(report_file) as f:
                report = json.load(f)

            # Verify summary contains total_duration instead of average_duration
            assert "summary" in report
            assert "total_duration" in report["summary"], (
                "Summary should contain total_duration"
            )
            assert report["summary"]["total_duration"] == 180.0, (
                "Total duration should be 60+120=180"
            )
            assert "average_duration" not in report["summary"], (
                "Summary should not contain average_duration"
            )

    def test_summary_total_duration_with_critic_files(self) -> None:
        """Test that total_duration is summed across critic files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create main output.jsonl file
            output_file = tmpdir_path / "output.jsonl"
            main_entries = [
                create_test_entry("test-1", cost=1.0, duration_seconds=60),
            ]
            with open(output_file, "w") as f:
                for entry in main_entries:
                    f.write(json.dumps(entry) + "\n")

            # Create critic attempt files
            critic_file_1 = tmpdir_path / "output.critic_attempt_1.jsonl"
            critic_entries_1 = [
                create_test_entry("test-1", cost=2.0, duration_seconds=100),
            ]
            with open(critic_file_1, "w") as f:
                for entry in critic_entries_1:
                    f.write(json.dumps(entry) + "\n")

            critic_file_2 = tmpdir_path / "output.critic_attempt_2.jsonl"
            critic_entries_2 = [
                create_test_entry("test-1", cost=3.0, duration_seconds=200),
            ]
            with open(critic_file_2, "w") as f:
                for entry in critic_entries_2:
                    f.write(json.dumps(entry) + "\n")

            # Import and run calculate_costs
            from benchmarks.utils.report_costs import calculate_costs

            calculate_costs(str(tmpdir_path))

            # Read the generated cost_report.jsonl
            report_file = tmpdir_path / "cost_report.jsonl"
            with open(report_file) as f:
                report = json.load(f)

            # When critic files exist, total_duration should be sum of critic files
            assert report["summary"]["total_duration"] == 300.0, (
                "Total duration should be 100+200=300 (from critic files)"
            )
            assert report["summary"]["total_cost"] == 5.0, (
                "Total cost should be 2+3=5 (from critic files)"
            )
