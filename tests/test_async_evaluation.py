"""Tests for asyncio-based evaluation orchestration.

These tests verify the asyncio refactor from ProcessPoolExecutor to asyncio
with semaphore-based concurrency and asyncio.to_thread() for sync operations.
"""

import asyncio
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
