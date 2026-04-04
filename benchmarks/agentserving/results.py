from __future__ import annotations

import math
import re
from statistics import mean
from typing import Iterable

from pydantic import BaseModel, Field


_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?)$"
)
_LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^\\"])*)"')


class HistogramSnapshot(BaseModel):
    sum: float = 0.0
    count: float = 0.0
    buckets: dict[float, float] = Field(default_factory=dict)


class PrometheusSnapshot(BaseModel):
    counters: dict[str, float] = Field(default_factory=dict)
    histograms: dict[str, HistogramSnapshot] = Field(default_factory=dict)


class HistogramSummary(BaseModel):
    count: int = 0
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None


class ServerMetricsSummary(BaseModel):
    request_success: float = 0.0
    prompt_tokens: float = 0.0
    generation_tokens: float = 0.0
    request_throughput: float | None = None
    prompt_token_throughput: float | None = None
    generation_token_throughput: float | None = None
    total_token_throughput: float | None = None
    ttft_seconds: HistogramSummary = Field(default_factory=HistogramSummary)
    e2e_latency_seconds: HistogramSummary = Field(default_factory=HistogramSummary)
    queue_time_seconds: HistogramSummary = Field(default_factory=HistogramSummary)
    inter_token_latency_seconds: HistogramSummary = Field(
        default_factory=HistogramSummary
    )


class AgentRunResult(BaseModel):
    agent_index: int
    workspace_dir: str
    persistence_dir: str
    started_at: str
    finished_at: str
    wall_clock_seconds: float
    execution_status: str
    used_finish_tool: bool
    finish_message: str | None = None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_call_count: int = 0
    first_response_latency_seconds: float | None = None
    mean_response_latency_seconds: float | None = None
    p95_response_latency_seconds: float | None = None
    accumulated_cost: float = 0.0
    html_output_exists: bool = False
    html_file_count: int = 0
    success: bool = False


class ExperimentSummary(BaseModel):
    parallelism: int
    agent_count: int
    success_count: int
    failure_count: int
    success_rate: float
    batch_wall_clock_seconds: float
    mean_agent_wall_clock_seconds: float | None = None
    p50_agent_wall_clock_seconds: float | None = None
    p95_agent_wall_clock_seconds: float | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    mean_first_response_latency_seconds: float | None = None
    mean_response_latency_seconds: float | None = None
    p95_response_latency_seconds: float | None = None
    prompt_token_throughput: float | None = None
    completion_token_throughput: float | None = None
    total_token_throughput: float | None = None
    server_metrics: ServerMetricsSummary | None = None
    collapsed: bool = False


class ExperimentConfig(BaseModel):
    model: str
    model_label: str
    machine_size: str
    context_length_k: int
    max_iterations: int
    agent_timeout_seconds: int
    task_prompt: str
    metrics_url: str | None = None


class ExperimentRecord(BaseModel):
    config: ExperimentConfig
    summary: ExperimentSummary
    results: list[AgentRunResult]


class SweepRecord(BaseModel):
    config: ExperimentConfig
    experiments: list[ExperimentSummary]


def parse_prometheus_snapshot(text: str) -> PrometheusSnapshot:
    snapshot = PrometheusSnapshot()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_RE.match(line)
        if match is None:
            continue
        name = match.group("name")
        labels = _parse_labels(match.group("labels") or "")
        value = float(match.group("value"))

        if name.endswith("_bucket"):
            base_name = name[: -len("_bucket")]
            histogram = snapshot.histograms.setdefault(base_name, HistogramSnapshot())
            le = _parse_bucket_bound(labels.get("le", "+Inf"))
            histogram.buckets[le] = histogram.buckets.get(le, 0.0) + value
        elif name.endswith("_sum"):
            base_name = name[: -len("_sum")]
            histogram = snapshot.histograms.setdefault(base_name, HistogramSnapshot())
            histogram.sum += value
        elif name.endswith("_count"):
            base_name = name[: -len("_count")]
            histogram = snapshot.histograms.setdefault(base_name, HistogramSnapshot())
            histogram.count += value
        else:
            snapshot.counters[name] = snapshot.counters.get(name, 0.0) + value
    return snapshot


def diff_prometheus_snapshots(
    before: PrometheusSnapshot | None,
    after: PrometheusSnapshot | None,
) -> PrometheusSnapshot | None:
    if before is None or after is None:
        return None

    diff = PrometheusSnapshot()
    counter_names = set(before.counters) | set(after.counters)
    for name in counter_names:
        value = after.counters.get(name, 0.0) - before.counters.get(name, 0.0)
        if value > 0:
            diff.counters[name] = value

    histogram_names = set(before.histograms) | set(after.histograms)
    for name in histogram_names:
        before_hist = before.histograms.get(name, HistogramSnapshot())
        after_hist = after.histograms.get(name, HistogramSnapshot())
        hist_diff = HistogramSnapshot(
            sum=max(0.0, after_hist.sum - before_hist.sum),
            count=max(0.0, after_hist.count - before_hist.count),
        )
        bucket_bounds = set(before_hist.buckets) | set(after_hist.buckets)
        for bound in bucket_bounds:
            value = after_hist.buckets.get(bound, 0.0) - before_hist.buckets.get(
                bound, 0.0
            )
            if value > 0:
                hist_diff.buckets[bound] = value
        if hist_diff.count > 0 or hist_diff.sum > 0 or hist_diff.buckets:
            diff.histograms[name] = hist_diff
    return diff


def summarize_histogram(histogram: HistogramSnapshot | None) -> HistogramSummary:
    if histogram is None or histogram.count <= 0:
        return HistogramSummary()
    return HistogramSummary(
        count=int(histogram.count),
        mean=histogram.sum / histogram.count,
        p50=_histogram_quantile(histogram, 0.50),
        p95=_histogram_quantile(histogram, 0.95),
    )


def summarize_server_metrics(
    metric_diff: PrometheusSnapshot | None,
    wall_clock_seconds: float,
) -> ServerMetricsSummary | None:
    if metric_diff is None:
        return None

    request_success = metric_diff.counters.get("vllm:request_success", 0.0)
    prompt_tokens = metric_diff.counters.get("vllm:prompt_tokens", 0.0)
    generation_tokens = metric_diff.counters.get("vllm:generation_tokens", 0.0)
    duration = wall_clock_seconds if wall_clock_seconds > 0 else None

    return ServerMetricsSummary(
        request_success=request_success,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        request_throughput=(request_success / duration) if duration else None,
        prompt_token_throughput=(prompt_tokens / duration) if duration else None,
        generation_token_throughput=(generation_tokens / duration)
        if duration
        else None,
        total_token_throughput=((prompt_tokens + generation_tokens) / duration)
        if duration
        else None,
        ttft_seconds=summarize_histogram(
            metric_diff.histograms.get("vllm:time_to_first_token_seconds")
        ),
        e2e_latency_seconds=summarize_histogram(
            metric_diff.histograms.get("vllm:e2e_request_latency_seconds")
        ),
        queue_time_seconds=summarize_histogram(
            metric_diff.histograms.get("vllm:request_queue_time_seconds")
        ),
        inter_token_latency_seconds=summarize_histogram(
            metric_diff.histograms.get("vllm:inter_token_latency_seconds")
        ),
    )


def summarize_samples(
    values: Iterable[float],
) -> tuple[float | None, float | None, float | None]:
    ordered = sorted(values)
    if not ordered:
        return None, None, None
    return mean(ordered), _percentile(ordered, 0.50), _percentile(ordered, 0.95)


def _percentile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = quantile * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    lower_value = values[lower]
    upper_value = values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def _histogram_quantile(histogram: HistogramSnapshot, quantile: float) -> float | None:
    if histogram.count <= 0:
        return None
    threshold = histogram.count * quantile
    for bound, cumulative_count in sorted(histogram.buckets.items()):
        if cumulative_count >= threshold:
            return None if math.isinf(bound) else bound
    return None


def _parse_labels(raw_labels: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for key, value in _LABEL_RE.findall(raw_labels):
        labels[key] = value.replace(r"\"", '"')
    return labels


def _parse_bucket_bound(bound: str) -> float:
    if bound == "+Inf":
        return math.inf
    return float(bound)
