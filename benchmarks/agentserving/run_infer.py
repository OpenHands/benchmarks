from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, cast
from urllib.parse import urlparse, urlunparse

import httpx

from benchmarks.agentserving.config import (
    DEFAULT_AGENT_TIMEOUT_SECONDS,
    DEFAULT_COLLAPSE_FAILURE_RATE,
    DEFAULT_MAX_FAKE_RESPONSES,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARALLELISM_LEVELS,
    DEFAULT_TASK_PROMPT,
    DEFAULT_WORKSPACE_ROOT,
)
from benchmarks.agentserving.results import (
    AgentRunResult,
    ExperimentConfig,
    ExperimentRecord,
    ExperimentSummary,
    PrometheusSnapshot,
    ServerMetricsSummary,
    SweepRecord,
    diff_prometheus_snapshots,
    parse_prometheus_snapshot,
    summarize_samples,
    summarize_server_metrics,
)
from benchmarks.utils.conversation import build_event_persistence_callback
from benchmarks.utils.evaluation_utils import construct_eval_output_dir
from benchmarks.utils.fake_user_response import fake_user_response
from benchmarks.utils.litellm_proxy import build_eval_llm
from benchmarks.utils.llm_config import load_llm_config
from openhands.sdk import LLM, Agent, Conversation, Event, LocalConversation, get_logger
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.tool.builtins.finish import FinishAction
from openhands.tools.preset.default import get_default_tools


logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile local-runtime OpenHands agents against an LLM endpoint"
    )
    parser.add_argument("llm_config_path", type=str, help="Path to JSON LLM config")
    parser.add_argument(
        "--parallelism-levels",
        type=int,
        nargs="+",
        default=DEFAULT_PARALLELISM_LEVELS,
        help="Parallel agent counts to run sequentially",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory for benchmark outputs",
    )
    parser.add_argument("--note", type=str, default=None, help="Optional run note")
    parser.add_argument(
        "--machine-size",
        type=str,
        default="4xh100",
        help="Label for the serving hardware configuration",
    )
    parser.add_argument(
        "--context-length-k",
        type=int,
        default=32,
        help="Label for the server max context length in thousands of tokens",
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default=None,
        help="Human-readable model label for reports",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Maximum OpenHands iterations per agent",
    )
    parser.add_argument(
        "--agent-timeout-seconds",
        type=int,
        default=DEFAULT_AGENT_TIMEOUT_SECONDS,
        help="Per-agent wall clock timeout",
    )
    parser.add_argument(
        "--max-fake-responses",
        type=int,
        default=DEFAULT_MAX_FAKE_RESPONSES,
        help="Maximum fake user responses before forcing the run to stop",
    )
    parser.add_argument(
        "--workspace-root",
        type=str,
        default=DEFAULT_WORKSPACE_ROOT,
        help="Parent directory used for random temporary workspaces",
    )
    parser.add_argument(
        "--task-prompt",
        type=str,
        default=DEFAULT_TASK_PROMPT,
        help="Prompt template. Supports {workspace_dir} substitution.",
    )
    parser.add_argument(
        "--metrics-url",
        type=str,
        default=None,
        help="Explicit vLLM /metrics endpoint. Defaults to deriving it from the LLM base_url.",
    )
    parser.add_argument(
        "--skip-healthcheck",
        action="store_true",
        help="Skip the pre-run /health probe",
    )
    parser.add_argument(
        "--enable-condenser",
        action="store_true",
        help="Enable the default LLM summarizing condenser",
    )
    parser.add_argument(
        "--show-trajectory",
        action="store_true",
        help="Print per-event trajectory lines to stdout while the benchmark runs",
    )
    parser.add_argument(
        "--keep-workspaces",
        action="store_true",
        help="Retain successful /tmp workspaces after the run",
    )
    parser.add_argument(
        "--cleanup-failed-workspaces",
        action="store_true",
        help="Also delete failed /tmp workspaces after the run",
    )
    parser.add_argument(
        "--collapse-failure-rate",
        type=float,
        default=DEFAULT_COLLAPSE_FAILURE_RATE,
        help="Stop the sweep once failure_rate >= this threshold at a parallelism level",
    )
    parser.add_argument(
        "--continue-after-collapse",
        action="store_true",
        help="Continue testing larger parallelism levels after a collapse is detected",
    )
    return parser


def infer_metrics_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    metrics_path = f"{path}/metrics" if path else "/metrics"
    return urlunparse(
        parsed._replace(path=metrics_path, params="", query="", fragment="")
    )


def infer_health_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    health_path = f"{path}/health" if path else "/health"
    return urlunparse(
        parsed._replace(path=health_path, params="", query="", fragment="")
    )


def scrape_metrics(metrics_url: str | None) -> PrometheusSnapshot | None:
    if metrics_url is None:
        return None
    response = httpx.get(metrics_url, timeout=30.0)
    response.raise_for_status()
    return parse_prometheus_snapshot(response.text)


def wait_for_health(health_url: str | None) -> None:
    if health_url is None:
        return
    response = httpx.get(health_url, timeout=30.0)
    response.raise_for_status()


def build_task_prompt(task_prompt: str, workspace_dir: str) -> str:
    return task_prompt.format(workspace_dir=workspace_dir)


def extract_finish_message(events: Sequence[Event]) -> tuple[bool, str | None]:
    for event in reversed(events):
        if isinstance(event, ActionEvent):
            action = event.action
            if isinstance(action, FinishAction):
                return True, action.message
            return False, None
    return False, None


def run_conversation_with_timeout(
    conversation: LocalConversation,
    *,
    timeout_seconds: float,
    pause_grace_seconds: float = 60.0,
) -> None:
    run_error: list[BaseException | None] = [None]
    finished = threading.Event()

    def _run() -> None:
        try:
            conversation.run()
        except BaseException as exc:  # pragma: no cover - re-raised below
            run_error[0] = exc
        finally:
            finished.set()

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    if not finished.wait(timeout_seconds):
        conversation.pause()
        if not finished.wait(pause_grace_seconds):
            raise TimeoutError(
                "Conversation exceeded the allotted runtime and did not pause within "
                f"{pause_grace_seconds:.0f}s"
            )
        raise TimeoutError(
            f"Conversation exceeded the allotted runtime of {timeout_seconds:.1f}s"
        )

    worker.join(timeout=0.1)
    if run_error[0] is not None:
        raise run_error[0]


def run_conversation_with_benchmark_fake_user_response(
    conversation: LocalConversation,
    *,
    timeout_seconds: int,
    max_fake_responses: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    fake_response_count = 0
    while True:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError(
                f"Conversation exceeded the allotted runtime of {timeout_seconds}s"
            )

        run_conversation_with_timeout(
            conversation,
            timeout_seconds=remaining_seconds,
        )
        status = conversation.state.execution_status
        if status != ConversationExecutionStatus.FINISHED:
            break

        events = list(conversation.state.events)
        used_finish_tool, _ = extract_finish_message(events)
        if used_finish_tool:
            break

        if not _agent_sent_message(events):
            break

        if fake_response_count >= max_fake_responses:
            break

        conversation.send_message(fake_user_response(conversation))
        fake_response_count += 1


async def run_parallelism_level(
    *,
    parallelism: int,
    llm: LLM,
    args: argparse.Namespace,
    experiment_dir: Path,
) -> ExperimentRecord:
    metrics_before = scrape_metrics(args.metrics_url)
    batch_start = time.perf_counter()

    loop = asyncio.get_running_loop()
    loop.set_default_executor(
        ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="agentserving")
    )
    semaphore = asyncio.Semaphore(parallelism)

    async def _run_one(agent_index: int) -> AgentRunResult:
        async with semaphore:
            return await asyncio.to_thread(
                execute_agent_run,
                agent_index,
                parallelism,
                llm,
                args,
                experiment_dir,
            )

    tasks = [
        asyncio.create_task(_run_one(agent_index)) for agent_index in range(parallelism)
    ]
    results: list[AgentRunResult] = []
    for task in asyncio.as_completed(tasks):
        results.append(await task)

    batch_wall_clock_seconds = time.perf_counter() - batch_start
    metrics_after = scrape_metrics(args.metrics_url)
    metric_diff = diff_prometheus_snapshots(metrics_before, metrics_after)
    server_metrics = summarize_server_metrics(metric_diff, batch_wall_clock_seconds)
    summary = build_experiment_summary(
        results=results,
        parallelism=parallelism,
        batch_wall_clock_seconds=batch_wall_clock_seconds,
        server_metrics=server_metrics,
        collapse_failure_rate=args.collapse_failure_rate,
    )

    config = ExperimentConfig(
        model=llm.model,
        model_label=args.model_label,
        machine_size=args.machine_size,
        context_length_k=args.context_length_k,
        max_iterations=args.max_iterations,
        agent_timeout_seconds=args.agent_timeout_seconds,
        task_prompt=args.task_prompt,
        metrics_url=args.metrics_url,
    )
    return ExperimentRecord(
        config=config,
        summary=summary,
        results=sorted(results, key=lambda item: item.agent_index),
    )


def execute_agent_run(
    agent_index: int,
    parallelism: int,
    llm: LLM,
    args: argparse.Namespace,
    experiment_dir: Path,
) -> AgentRunResult:
    workspace_dir = Path(
        tempfile.mkdtemp(
            prefix=f"agentserving-p{parallelism:02d}-a{agent_index:02d}-",
            dir=args.workspace_root,
        )
    )
    persistence_dir = experiment_dir / "conversations" / f"agent_{agent_index:03d}"
    persistence_dir.mkdir(parents=True, exist_ok=True)
    instance_id = f"p{parallelism:02d}-a{agent_index:02d}"

    run_llm = build_eval_llm(llm, usage_id=f"agentserving-{instance_id}")
    condenser = None
    if args.enable_condenser:
        condenser = LLMSummarizingCondenser(
            llm=build_eval_llm(llm, usage_id=f"agentserving-{instance_id}-condenser"),
            max_size=80,
            keep_first=4,
        )
    agent = Agent(
        llm=run_llm,
        tools=get_default_tools(enable_browser=False),
        system_prompt_kwargs={"cli_mode": True},
        condenser=condenser,
    )
    callback = build_event_persistence_callback(
        run_id=str(experiment_dir),
        instance_id=instance_id,
        show_trajectory=args.show_trajectory,
    )
    conversation: LocalConversation = cast(
        LocalConversation,
        Conversation(
            agent=agent,
            workspace=workspace_dir,
            persistence_dir=persistence_dir,
            callbacks=[callback],
            max_iteration_per_run=args.max_iterations,
            delete_on_close=False,
        ),
    )

    started_at = datetime.now(timezone.utc)
    error: str | None = None
    try:
        conversation.send_message(
            build_task_prompt(args.task_prompt, str(workspace_dir))
        )
        run_conversation_with_benchmark_fake_user_response(
            conversation,
            timeout_seconds=args.agent_timeout_seconds,
            max_fake_responses=args.max_fake_responses,
        )
    except Exception as exc:
        error = str(exc)
    finally:
        finished_at = datetime.now(timezone.utc)
        wall_clock_seconds = (finished_at - started_at).total_seconds()
        metrics = conversation.conversation_stats.get_combined_metrics()
        latencies = [item.latency for item in metrics.response_latencies]
        _, _, response_latency_p95 = summarize_samples(latencies)
        used_finish_tool, finish_message = extract_finish_message(
            list(conversation.state.events)
        )
        html_file_count = len(list(workspace_dir.rglob("*.html")))
        html_output_exists = (workspace_dir / "site" / "index.html").is_file()
        prompt_tokens = 0
        completion_tokens = 0
        if metrics.accumulated_token_usage is not None:
            prompt_tokens = metrics.accumulated_token_usage.prompt_tokens
            completion_tokens = metrics.accumulated_token_usage.completion_tokens
        execution_status = conversation.state.execution_status.value
        conversation.close()

    success = (
        error is None
        and execution_status == ConversationExecutionStatus.FINISHED.value
        and used_finish_tool
    )

    should_cleanup = (not args.keep_workspaces and success) or (
        args.cleanup_failed_workspaces and not success
    )
    if should_cleanup:
        shutil.rmtree(workspace_dir, ignore_errors=True)

    return AgentRunResult(
        agent_index=agent_index,
        workspace_dir=str(workspace_dir),
        persistence_dir=str(persistence_dir),
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        wall_clock_seconds=wall_clock_seconds,
        execution_status=execution_status,
        used_finish_tool=used_finish_tool,
        finish_message=finish_message,
        error=error,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        llm_call_count=len(metrics.response_latencies),
        first_response_latency_seconds=latencies[0] if latencies else None,
        mean_response_latency_seconds=(sum(latencies) / len(latencies))
        if latencies
        else None,
        p95_response_latency_seconds=response_latency_p95,
        accumulated_cost=metrics.accumulated_cost,
        html_output_exists=html_output_exists,
        html_file_count=html_file_count,
        success=success,
    )


def build_experiment_summary(
    *,
    results: Sequence[AgentRunResult],
    parallelism: int,
    batch_wall_clock_seconds: float,
    server_metrics: ServerMetricsSummary | None,
    collapse_failure_rate: float,
) -> ExperimentSummary:
    success_count = sum(1 for result in results if result.success)
    failure_count = len(results) - success_count
    success_rate = success_count / len(results) if results else 0.0
    durations = [result.wall_clock_seconds for result in results]
    mean_duration, p50_duration, p95_duration = summarize_samples(durations)
    first_latencies = [
        result.first_response_latency_seconds
        for result in results
        if result.first_response_latency_seconds is not None
    ]
    response_latencies = [
        result.mean_response_latency_seconds
        for result in results
        if result.mean_response_latency_seconds is not None
    ]
    p95_latencies = [
        result.p95_response_latency_seconds
        for result in results
        if result.p95_response_latency_seconds is not None
    ]
    mean_first_latency, _, _ = summarize_samples(first_latencies)
    mean_response_latency, _, _ = summarize_samples(response_latencies)
    _, _, p95_response_latency = summarize_samples(p95_latencies)
    total_prompt_tokens = sum(result.prompt_tokens for result in results)
    total_completion_tokens = sum(result.completion_tokens for result in results)
    total_cost = sum(result.accumulated_cost for result in results)
    duration = batch_wall_clock_seconds if batch_wall_clock_seconds > 0 else None
    prompt_token_throughput = (
        total_prompt_tokens / duration if duration is not None else None
    )
    completion_token_throughput = (
        total_completion_tokens / duration if duration is not None else None
    )
    total_token_throughput = (
        (total_prompt_tokens + total_completion_tokens) / duration
        if duration is not None
        else None
    )
    collapsed = (
        failure_count > 0 and failure_count / len(results) >= collapse_failure_rate
    )

    if server_metrics is not None and (
        server_metrics.prompt_tokens > 0
        or server_metrics.generation_tokens > 0
        or server_metrics.request_success > 0
    ):
        prompt_token_throughput = server_metrics.prompt_token_throughput
        completion_token_throughput = server_metrics.generation_token_throughput
        total_token_throughput = server_metrics.total_token_throughput

    return ExperimentSummary(
        parallelism=parallelism,
        agent_count=len(results),
        success_count=success_count,
        failure_count=failure_count,
        success_rate=success_rate,
        batch_wall_clock_seconds=batch_wall_clock_seconds,
        mean_agent_wall_clock_seconds=mean_duration,
        p50_agent_wall_clock_seconds=p50_duration,
        p95_agent_wall_clock_seconds=p95_duration,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_cost=total_cost,
        mean_first_response_latency_seconds=mean_first_latency,
        mean_response_latency_seconds=mean_response_latency,
        p95_response_latency_seconds=p95_response_latency,
        prompt_token_throughput=prompt_token_throughput,
        completion_token_throughput=completion_token_throughput,
        total_token_throughput=total_token_throughput,
        server_metrics=server_metrics,
        collapsed=collapsed,
    )


def write_experiment_record(experiment_dir: Path, record: ExperimentRecord) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "experiment.json").write_text(
        record.model_dump_json(indent=2),
        encoding="utf-8",
    )
    with (experiment_dir / "results.jsonl").open("w", encoding="utf-8") as file_obj:
        for result in record.results:
            file_obj.write(result.model_dump_json() + "\n")


def _agent_sent_message(events: Sequence[Event]) -> bool:
    for event in reversed(events):
        if isinstance(event, MessageEvent) and event.source == "agent":
            return True
        if isinstance(event, ActionEvent):
            return False
    return False


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    llm = load_llm_config(args.llm_config_path)
    args.model_label = args.model_label or llm.model
    args.metrics_url = args.metrics_url or infer_metrics_url(llm.base_url)

    eval_output_dir = Path(
        construct_eval_output_dir(
            args.output_dir,
            "agentserving",
            llm.model.replace("/", "__"),
            args.max_iterations,
            args.note,
        )
    )
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_healthcheck:
        wait_for_health(infer_health_url(llm.base_url))

    logger.info("Starting agentserving benchmark")
    logger.info("Output directory: %s", eval_output_dir)
    logger.info("Parallelism levels: %s", args.parallelism_levels)
    logger.info("Metrics URL: %s", args.metrics_url)

    experiment_summaries: list[ExperimentSummary] = []
    config = ExperimentConfig(
        model=llm.model,
        model_label=args.model_label,
        machine_size=args.machine_size,
        context_length_k=args.context_length_k,
        max_iterations=args.max_iterations,
        agent_timeout_seconds=args.agent_timeout_seconds,
        task_prompt=args.task_prompt,
        metrics_url=args.metrics_url,
    )

    for parallelism in args.parallelism_levels:
        experiment_dir = eval_output_dir / f"parallelism_{parallelism:02d}"
        record = asyncio.run(
            run_parallelism_level(
                parallelism=parallelism,
                llm=llm,
                args=args,
                experiment_dir=experiment_dir,
            )
        )
        write_experiment_record(experiment_dir, record)
        experiment_summaries.append(record.summary)
        logger.info(
            "Parallelism=%d success=%d/%d batch_wall_clock=%.2fs",
            parallelism,
            record.summary.success_count,
            record.summary.agent_count,
            record.summary.batch_wall_clock_seconds,
        )
        if record.summary.collapsed and not args.continue_after_collapse:
            logger.warning(
                "Detected collapse at parallelism=%d (failure_rate=%.2f); stopping sweep",
                parallelism,
                1.0 - record.summary.success_rate,
            )
            break

    sweep = SweepRecord(config=config, experiments=experiment_summaries)
    (eval_output_dir / "sweep.json").write_text(
        sweep.model_dump_json(indent=2),
        encoding="utf-8",
    )
    with (eval_output_dir / "sweep.jsonl").open("w", encoding="utf-8") as file_obj:
        for summary in experiment_summaries:
            file_obj.write(summary.model_dump_json() + "\n")
    print(json.dumps(sweep.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
