from __future__ import annotations

from benchmarks.agentserving.results import (
    diff_prometheus_snapshots,
    parse_prometheus_snapshot,
    summarize_server_metrics,
)
from benchmarks.agentserving.run_infer import infer_metrics_url


BEFORE = """
# TYPE vllm:request_success counter
vllm:request_success{model_name=\"demo\"} 10
vllm:prompt_tokens{model_name=\"demo\"} 1000
vllm:generation_tokens{model_name=\"demo\"} 500
vllm:time_to_first_token_seconds_bucket{le=\"0.5\",model_name=\"demo\"} 3
vllm:time_to_first_token_seconds_bucket{le=\"1.0\",model_name=\"demo\"} 5
vllm:time_to_first_token_seconds_bucket{le=\"+Inf\",model_name=\"demo\"} 5
vllm:time_to_first_token_seconds_sum{model_name=\"demo\"} 3.2
vllm:time_to_first_token_seconds_count{model_name=\"demo\"} 5
vllm:e2e_request_latency_seconds_bucket{le=\"2.0\",model_name=\"demo\"} 5
vllm:e2e_request_latency_seconds_bucket{le=\"+Inf\",model_name=\"demo\"} 5
vllm:e2e_request_latency_seconds_sum{model_name=\"demo\"} 7.0
vllm:e2e_request_latency_seconds_count{model_name=\"demo\"} 5
"""

AFTER = """
vllm:request_success{model_name=\"demo\"} 16
vllm:prompt_tokens{model_name=\"demo\"} 1600
vllm:generation_tokens{model_name=\"demo\"} 860
vllm:time_to_first_token_seconds_bucket{le=\"0.5\",model_name=\"demo\"} 5
vllm:time_to_first_token_seconds_bucket{le=\"1.0\",model_name=\"demo\"} 10
vllm:time_to_first_token_seconds_bucket{le=\"+Inf\",model_name=\"demo\"} 11
vllm:time_to_first_token_seconds_sum{model_name=\"demo\"} 7.6
vllm:time_to_first_token_seconds_count{model_name=\"demo\"} 11
vllm:e2e_request_latency_seconds_bucket{le=\"2.0\",model_name=\"demo\"} 8
vllm:e2e_request_latency_seconds_bucket{le=\"+Inf\",model_name=\"demo\"} 11
vllm:e2e_request_latency_seconds_sum{model_name=\"demo\"} 16.0
vllm:e2e_request_latency_seconds_count{model_name=\"demo\"} 11
"""


def test_infer_metrics_url_strips_v1_suffix() -> None:
    assert (
        infer_metrics_url("https://example.modal.run/v1")
        == "https://example.modal.run/metrics"
    )
    assert (
        infer_metrics_url("https://example.modal.run/custom/prefix/v1")
        == "https://example.modal.run/custom/prefix/metrics"
    )


def test_prometheus_diff_and_server_summary() -> None:
    before = parse_prometheus_snapshot(BEFORE)
    after = parse_prometheus_snapshot(AFTER)
    diff = diff_prometheus_snapshots(before, after)
    assert diff is not None

    server_metrics = summarize_server_metrics(diff, wall_clock_seconds=20.0)
    assert server_metrics is not None
    assert server_metrics.request_success == 6
    assert server_metrics.prompt_tokens == 600
    assert server_metrics.generation_tokens == 360
    assert server_metrics.request_throughput == 0.3
    assert server_metrics.ttft_seconds.count == 6
    assert server_metrics.ttft_seconds.mean is not None
    assert abs(server_metrics.ttft_seconds.mean - (4.4 / 6)) < 1e-9
    assert server_metrics.ttft_seconds.p50 == 1.0
    assert server_metrics.e2e_latency_seconds.count == 6
    assert server_metrics.e2e_latency_seconds.mean is not None
    assert abs(server_metrics.e2e_latency_seconds.mean - 1.5) < 1e-9
