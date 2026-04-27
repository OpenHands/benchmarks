"""Tests for cost report generation."""

import json

from benchmarks.utils.report_costs import calculate_costs, extract_proxy_cost


def test_extract_proxy_cost_sums_entries():
    jsonl_data = [
        {"test_result": {"proxy_cost": 1.25}},
        {"test_result": {"proxy_cost": 2.5}},
        {"test_result": {}},
        None,
    ]

    total, zero_count = extract_proxy_cost(jsonl_data)
    assert total == 3.75
    assert zero_count == 0


def test_extract_proxy_cost_counts_zero_cost_instances():
    jsonl_data = [
        {"test_result": {"proxy_cost": 1.25}},
        {"test_result": {"proxy_cost": 0.0}},
        {"test_result": {"proxy_cost": 0.0}},
        {"test_result": {}},
    ]

    total, zero_count = extract_proxy_cost(jsonl_data)
    assert total == 1.25
    assert zero_count == 2


def test_calculate_costs_writes_top_level_proxy_cost_summary(tmp_path):
    output_file = tmp_path / "output.jsonl"
    critic_file = tmp_path / "output.critic_attempt_1.jsonl"

    output_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "metrics": {"accumulated_cost": 1.0},
                        "test_result": {"proxy_cost": 2.0},
                        "history": [],
                    }
                ),
                json.dumps(
                    {
                        "metrics": {"accumulated_cost": 3.0},
                        "test_result": {"proxy_cost": 4.0},
                        "history": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    critic_file.write_text(
        json.dumps(
            {
                "metrics": {"accumulated_cost": 5.0},
                "test_result": {"proxy_cost": 6.0},
                "history": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    calculate_costs(str(tmp_path))

    report = json.loads((tmp_path / "cost_report.jsonl").read_text(encoding="utf-8"))

    assert "proxy_cost_summary" in report
    assert report["proxy_cost_summary"]["total_proxy_cost"] == 6.0
    assert report["proxy_cost_summary"]["zero_proxy_cost_instances"] == 0
    assert report["proxy_cost_summary"]["only_main_output_proxy_cost"] == 6.0
    assert report["proxy_cost_summary"]["sum_critic_files_proxy_cost"] == 6.0
