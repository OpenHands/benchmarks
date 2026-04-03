"""Tests for asyncio-based evaluation orchestration.

These tests verify the asyncio refactor from ProcessPoolExecutor to asyncio
with semaphore-based concurrency and asyncio.to_thread() for sync operations.
"""

import asyncio
import concurrent.futures
import time

import pytest


def slow_worker_sync(instance_id: str, sleep_time: float) -> tuple[str, dict]:
    """Simulate a slow worker that takes sleep_time seconds (synchronous)."""
    time.sleep(sleep_time)
    return instance_id, {"status": "completed"}


@pytest.mark.asyncio
async def test_asyncio_semaphore_concurrency():
    """Test that asyncio.Semaphore correctly limits concurrent tasks."""
    max_workers = 2
    semaphore = asyncio.Semaphore(max_workers)
    concurrent_count = 0
    max_concurrent = 0

    async def track_concurrency(instance_id: str) -> str:
        nonlocal concurrent_count, max_concurrent
        async with semaphore:
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)  # Simulate work
            concurrent_count -= 1
            return instance_id

    # Create more tasks than workers
    tasks = [asyncio.create_task(track_concurrency(f"inst_{i}")) for i in range(6)]

    results = await asyncio.gather(*tasks)

    assert len(results) == 6, "All instances should complete"
    assert max_concurrent <= max_workers, (
        f"Should never exceed {max_workers} concurrent tasks, got {max_concurrent}"
    )


@pytest.mark.asyncio
async def test_asyncio_to_thread_runs_sync_code():
    """Test that asyncio.to_thread() correctly runs sync code."""

    def sync_work() -> str:
        time.sleep(0.1)  # Sync sleep - would block event loop if not in thread
        return "done"

    # Run multiple sync operations concurrently via to_thread
    results = await asyncio.gather(
        asyncio.to_thread(sync_work),
        asyncio.to_thread(sync_work),
        asyncio.to_thread(sync_work),
    )

    assert results == ["done", "done", "done"]


@pytest.mark.asyncio
async def test_asyncio_timeout_detection():
    """Test that per-instance timeouts are correctly detected with asyncio.wait."""
    instance_timeout = 0.5  # 500ms for test

    async def slow_task(instance_id: str, sleep_time: float) -> tuple[str, dict]:
        await asyncio.to_thread(time.sleep, sleep_time)
        return instance_id, {"status": "completed"}

    # Track pending instances
    pending_instances: dict[asyncio.Task, dict] = {}
    completed: list[str] = []
    timed_out: list[str] = []

    # Create tasks with different durations
    fast_task = asyncio.create_task(slow_task("fast", 0.1))
    slow_task1 = asyncio.create_task(slow_task("slow", 5.0))

    pending_instances[fast_task] = {
        "instance_id": "fast",
        "start_time": time.monotonic(),
    }
    pending_instances[slow_task1] = {
        "instance_id": "slow",
        "start_time": time.monotonic(),
    }

    pending = {fast_task, slow_task1}

    while pending:
        done, pending = await asyncio.wait(
            pending, timeout=0.2, return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            instance_id, _ = task.result()
            completed.append(instance_id)

        # Check for timeouts
        now = time.monotonic()
        timed_out_tasks = [
            task
            for task in pending
            if now - pending_instances[task]["start_time"] > instance_timeout
        ]

        for task in timed_out_tasks:
            pending.discard(task)
            timed_out.append(pending_instances[task]["instance_id"])
            task.cancel()

    assert "fast" in completed, "Fast instance should complete"
    assert "slow" in timed_out, "Slow instance should timeout"


@pytest.mark.asyncio
async def test_asyncio_wait_returns_completed_first():
    """Test that asyncio.wait with FIRST_COMPLETED returns done tasks first."""
    completed_order: list[str] = []

    async def ordered_task(instance_id: str, delay: float) -> str:
        await asyncio.sleep(delay)
        completed_order.append(instance_id)
        return instance_id

    task1 = asyncio.create_task(ordered_task("slow", 0.3))
    task2 = asyncio.create_task(ordered_task("fast", 0.1))
    task3 = asyncio.create_task(ordered_task("medium", 0.2))

    pending = {task1, task2, task3}
    results = []

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            results.append(task.result())

    # Results should be in completion order
    assert completed_order == ["fast", "medium", "slow"]


@pytest.mark.asyncio
async def test_asyncio_task_cancellation():
    """Test that cancelled tasks raise CancelledError."""
    cancelled = False

    async def cancellable_task() -> str:
        nonlocal cancelled
        try:
            await asyncio.sleep(10)  # Long sleep
            return "completed"
        except asyncio.CancelledError:
            cancelled = True
            raise

    task = asyncio.create_task(cancellable_task())
    await asyncio.sleep(0.1)  # Let task start
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled, "Task should have been cancelled"


@pytest.mark.asyncio
async def test_concurrent_file_write_with_lock():
    """Test that asyncio.Lock prevents concurrent file write issues."""
    write_count = 0
    lock = asyncio.Lock()

    async def write_with_lock(data: str) -> None:
        nonlocal write_count
        async with lock:
            # Simulate file write operation
            write_count += 1
            await asyncio.sleep(0.01)

    # Multiple concurrent writes
    await asyncio.gather(*[write_with_lock(f"data_{i}") for i in range(10)])

    assert write_count == 10, "All writes should complete"


def test_sync_wrapper():
    """Test that sync code can run asyncio via asyncio.run()."""

    async def async_work() -> int:
        await asyncio.sleep(0.01)
        return 42

    # This is how _run_iterative_mode calls _run_iterative_mode_async
    result = asyncio.run(async_work())
    assert result == 42


def test_evaluation_run_end_to_end(tmp_path):
    """Integration test: run a real Evaluation instance through the async path.

    Creates a TestEvaluation with mock workspaces and verifies that:
    - Multiple instances run concurrently via asyncio
    - Results are collected and written to attempt files
    - Errors produce error outputs (not lost instances)
    - Thread-safe logging is properly set up
    """
    from typing import List
    from unittest.mock import Mock, patch

    from benchmarks.utils.evaluation import Evaluation
    from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
    from openhands.sdk import LLM
    from openhands.sdk.critic import PassCritic

    instances = [EvalInstance(id=f"inst_{i}", data={"idx": i}) for i in range(4)]
    # Instance 2 will fail
    fail_ids = {"inst_2"}

    class TestEvaluation(Evaluation):
        def prepare_instances(self) -> List[EvalInstance]:
            return instances

        def prepare_workspace(self, instance, resource_factor=1, forward_env=None):
            ws = Mock()
            ws.__exit__ = Mock()
            return ws

        def evaluate_instance(self, instance, workspace):
            if instance.id in fail_ids:
                raise RuntimeError(f"Simulated failure for {instance.id}")
            return EvalOutput(
                instance_id=instance.id,
                test_result={"ok": True},
                instruction="test",
                error=None,
                history=[],
                instance=instance.data,
            )

    llm = LLM(model="test-model")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir=str(tmp_path),
        details={},
        eval_limit=4,
        n_critic_runs=1,
        max_retries=0,
        critic=PassCritic(),
    )

    evaluator = TestEvaluation(metadata=metadata, num_workers=2)

    with patch("benchmarks.utils.evaluation.LaminarService") as mock_lmnr:
        svc = Mock()
        svc.create_evaluation.return_value = None
        svc.create_evaluation_datapoint.return_value = None
        mock_lmnr.get.return_value = svc

        results = evaluator.run()

    # All 4 instances should produce output (3 success + 1 error)
    assert len(results) == 4
    result_ids = {r.instance_id for r in results}
    assert result_ids == {"inst_0", "inst_1", "inst_2", "inst_3"}

    # Check error output was created for the failing instance
    error_results = [r for r in results if r.error is not None]
    assert len(error_results) == 1
    assert error_results[0].instance_id == "inst_2"

    # Check attempt file was written
    attempt_file = tmp_path / "output.critic_attempt_1.jsonl"
    assert attempt_file.exists()
    lines = attempt_file.read_text().strip().split("\n")
    assert len(lines) == 4


def test_evaluation_installs_explicit_thread_executor(tmp_path, monkeypatch):
    """Evaluation should size asyncio.to_thread() from num_workers."""
    from typing import List
    from unittest.mock import Mock, patch

    from benchmarks.utils.evaluation import Evaluation
    from benchmarks.utils.models import EvalInstance, EvalMetadata
    from openhands.sdk import LLM
    from openhands.sdk.critic import PassCritic

    captured: dict[str, object] = {}

    class TestEvaluation(Evaluation):
        def prepare_instances(self) -> List[EvalInstance]:
            return []

        def prepare_workspace(self, instance, resource_factor=1, forward_env=None):
            raise AssertionError("prepare_workspace should not be called")

        def evaluate_instance(self, instance, workspace):
            raise AssertionError("evaluate_instance should not be called")

    real_executor = concurrent.futures.ThreadPoolExecutor

    def recording_executor(*args, **kwargs):
        captured["max_workers"] = kwargs["max_workers"]
        captured["thread_name_prefix"] = kwargs.get("thread_name_prefix")
        executor = real_executor(*args, **kwargs)
        captured["executor"] = executor
        return executor

    monkeypatch.setattr(
        "benchmarks.utils.evaluation.ThreadPoolExecutor",
        recording_executor,
    )

    llm = LLM(model="test-model")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir=str(tmp_path),
        details={},
        eval_limit=0,
        n_critic_runs=1,
        max_retries=0,
        critic=PassCritic(),
    )

    evaluator = TestEvaluation(metadata=metadata, num_workers=7)

    with patch("benchmarks.utils.evaluation.LaminarService") as mock_lmnr:
        svc = Mock()
        svc.create_evaluation.return_value = None
        mock_lmnr.get.return_value = svc

        results = evaluator.run()

    assert results == []
    assert captured["max_workers"] == 7
    assert captured["thread_name_prefix"] == "evaluation-worker"


def test_evaluation_logs_effective_executor_capacity(tmp_path, monkeypatch):
    """Startup logs should expose configured vs effective thread capacity."""
    from typing import List
    from unittest.mock import AsyncMock, Mock, patch

    from benchmarks.utils.evaluation import Evaluation, logger
    from benchmarks.utils.models import EvalInstance, EvalMetadata
    from openhands.sdk import LLM
    from openhands.sdk.critic import PassCritic

    class TestEvaluation(Evaluation):
        def prepare_instances(self) -> List[EvalInstance]:
            return [EvalInstance(id="inst-1", data={})]

        def prepare_workspace(self, instance, resource_factor=1, forward_env=None):
            raise AssertionError("prepare_workspace should not be called")

        def evaluate_instance(self, instance, workspace):
            raise AssertionError("evaluate_instance should not be called")

    monkeypatch.setattr("benchmarks.utils.evaluation.os.cpu_count", lambda: 4)
    monkeypatch.setattr(
        "benchmarks.utils.evaluation._read_cgroup_cpu_max",
        lambda: "400000 100000",
    )

    llm = LLM(model="test-model")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir=str(tmp_path),
        details={},
        eval_limit=0,
        n_critic_runs=1,
        max_retries=0,
        critic=PassCritic(),
    )

    evaluator = TestEvaluation(metadata=metadata, num_workers=30)

    with (
        patch("benchmarks.utils.evaluation.LaminarService") as mock_lmnr,
        patch.object(logger, "info") as mock_info,
    ):
        svc = Mock()
        svc.create_evaluation.return_value = None
        mock_lmnr.get.return_value = svc
        evaluator._run_attempt_async = AsyncMock(return_value=[])
        evaluator.run()

    assert any(
        call.args
        and call.args[0]
        == "[executor] configured_workers=%d executor_cap=%d effective_max_workers=%d default_max_workers=%d os_cpu_count=%s cpu.max=%s"
        and call.args[1:] == (30, 20, 20, 8, 4, "400000 100000")
        for call in mock_info.call_args_list
    )


def test_evaluation_caps_thread_executor_workers(tmp_path, monkeypatch):
    """Evaluation should cap asyncio.to_thread() workers to a configured maximum."""
    from typing import List
    from unittest.mock import Mock, patch

    from benchmarks.utils.evaluation import Evaluation, logger
    from benchmarks.utils.models import EvalInstance, EvalMetadata
    from openhands.sdk import LLM
    from openhands.sdk.critic import PassCritic

    captured: dict[str, object] = {}

    class TestEvaluation(Evaluation):
        def prepare_instances(self) -> List[EvalInstance]:
            return []

        def prepare_workspace(self, instance, resource_factor=1, forward_env=None):
            raise AssertionError("prepare_workspace should not be called")

        def evaluate_instance(self, instance, workspace):
            raise AssertionError("evaluate_instance should not be called")

    real_executor = concurrent.futures.ThreadPoolExecutor

    def recording_executor(*args, **kwargs):
        captured["max_workers"] = kwargs["max_workers"]
        return real_executor(*args, **kwargs)

    monkeypatch.setattr(
        "benchmarks.utils.evaluation.ThreadPoolExecutor",
        recording_executor,
    )

    llm = LLM(model="test-model")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir=str(tmp_path),
        details={},
        eval_limit=0,
        n_critic_runs=1,
        max_retries=0,
        critic=PassCritic(),
    )

    evaluator = TestEvaluation(
        metadata=metadata,
        num_workers=1000,
        max_asyncio_thread_workers=20,
    )

    with (
        patch("benchmarks.utils.evaluation.LaminarService") as mock_lmnr,
        patch.object(logger, "warning") as mock_warning,
    ):
        svc = Mock()
        svc.create_evaluation.return_value = None
        mock_lmnr.get.return_value = svc
        results = evaluator.run()

    assert results == []
    assert captured["max_workers"] == 20
    assert any(
        call.args
        and call.args[0]
        == "[executor] capping configured_workers=%d to executor_cap=%d"
        and call.args[1:] == (1000, 20)
        for call in mock_warning.call_args_list
    )


def test_evaluation_timeout_cancels_instance(tmp_path):
    """Integration test: verify that per-instance timeouts cancel instances."""
    from typing import List
    from unittest.mock import Mock, patch

    from benchmarks.utils.evaluation import Evaluation
    from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
    from openhands.sdk import LLM
    from openhands.sdk.critic import PassCritic

    class TestEvaluation(Evaluation):
        def prepare_instances(self) -> List[EvalInstance]:
            return [
                EvalInstance(id="fast", data={}),
                EvalInstance(id="slow", data={}),
            ]

        def prepare_workspace(self, instance, resource_factor=1, forward_env=None):
            ws = Mock()
            ws.__exit__ = Mock()
            return ws

        def evaluate_instance(self, instance, workspace):
            if instance.id == "slow":
                time.sleep(8)  # Will be cancelled by timeout
            return EvalOutput(
                instance_id=instance.id,
                test_result={"ok": True},
                instruction="test",
                error=None,
                history=[],
                instance=instance.data,
            )

    llm = LLM(model="test-model")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir=str(tmp_path),
        details={},
        eval_limit=2,
        n_critic_runs=1,
        max_retries=0,
        critic=PassCritic(),
    )

    evaluator = TestEvaluation(metadata=metadata, num_workers=2, instance_timeout=2)

    with (
        patch("benchmarks.utils.evaluation.LaminarService") as mock_lmnr,
        patch("benchmarks.utils.evaluation.TIMEOUT_CHECK_INTERVAL_SECONDS", 1),
    ):
        svc = Mock()
        svc.create_evaluation.return_value = None
        svc.create_evaluation_datapoint.return_value = None
        mock_lmnr.get.return_value = svc

        results = evaluator.run()

    result_ids = {r.instance_id for r in results}
    assert "fast" in result_ids
    assert "slow" in result_ids

    # The slow instance should have a timeout error
    slow_result = [r for r in results if r.instance_id == "slow"][0]
    assert slow_result.error is not None
    assert "timeout" in slow_result.error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
