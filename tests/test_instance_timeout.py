"""Tests for per-instance timeout handling in the evaluation module."""

import asyncio
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

import pytest


@dataclass
class MockPendingInstance:
    """Mock for testing the thread-pool queue time fix."""

    instance_id: str
    start_time: float | None = None


def slow_worker(instance_id: str, sleep_time: float) -> tuple[str, dict]:
    """Simulate a slow worker that takes sleep_time seconds."""
    time.sleep(sleep_time)
    return instance_id, {"status": "completed"}


def test_per_instance_timeout_logic():
    """Test that per-instance timeout logic correctly identifies timed-out futures."""
    instance_timeout = 0.1

    with ThreadPoolExecutor(max_workers=4) as pool:
        # Submit jobs with different durations
        futures = []
        future_to_instance = {}
        future_start_times = {}

        # Fast job (should complete)
        fut1 = pool.submit(slow_worker, "fast_instance", 0.01)
        futures.append(fut1)
        future_to_instance[fut1] = "fast_instance"
        future_start_times[fut1] = time.monotonic()

        # Slow job (should timeout)
        fut2 = pool.submit(slow_worker, "slow_instance", 0.5)
        futures.append(fut2)
        future_to_instance[fut2] = "slow_instance"
        future_start_times[fut2] = time.monotonic()

        pending = set(futures)
        completed = []
        timed_out = []

        while pending:
            done, pending = wait(pending, timeout=0.02, return_when=FIRST_COMPLETED)

            for fut in done:
                instance_id, result = fut.result()
                completed.append(instance_id)

            # Check for per-instance timeouts
            now = time.monotonic()
            timed_out_futures = [
                fut
                for fut in pending
                if now - future_start_times[fut] > instance_timeout
            ]

            for fut in timed_out_futures:
                pending.discard(fut)
                timed_out.append(future_to_instance[fut])
                fut.cancel()

        assert "fast_instance" in completed, "Fast instance should complete"
        assert "slow_instance" in timed_out, "Slow instance should timeout"
        assert "slow_instance" not in completed, "Slow instance should not complete"


def test_all_instances_complete_before_timeout():
    """Test that when all instances complete quickly, no timeouts occur."""
    instance_timeout = 1.0

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        future_to_instance = {}
        future_start_times = {}

        for i in range(3):
            fut = pool.submit(slow_worker, f"instance_{i}", 0.01)
            futures.append(fut)
            future_to_instance[fut] = f"instance_{i}"
            future_start_times[fut] = time.monotonic()

        pending = set(futures)
        completed = []
        timed_out = []

        while pending:
            done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)

            for fut in done:
                instance_id, result = fut.result()
                completed.append(instance_id)

            now = time.monotonic()
            timed_out_futures = [
                fut
                for fut in pending
                if now - future_start_times[fut] > instance_timeout
            ]

            for fut in timed_out_futures:
                pending.discard(fut)
                timed_out.append(future_to_instance[fut])
                fut.cancel()

        assert len(completed) == 3, "All instances should complete"
        assert len(timed_out) == 0, "No instances should timeout"


def test_multiple_timeouts():
    """Test that multiple instances can timeout independently."""
    instance_timeout = 0.1

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        future_to_instance = {}
        future_start_times = {}

        # Mix of fast and slow jobs
        configs = [
            ("fast_1", 0.01),
            ("slow_1", 0.5),
            ("fast_2", 0.02),
            ("slow_2", 0.5),
        ]

        for instance_id, sleep_time in configs:
            fut = pool.submit(slow_worker, instance_id, sleep_time)
            futures.append(fut)
            future_to_instance[fut] = instance_id
            future_start_times[fut] = time.monotonic()

        pending = set(futures)
        completed = []
        timed_out = []

        while pending:
            done, pending = wait(pending, timeout=0.02, return_when=FIRST_COMPLETED)

            for fut in done:
                instance_id, result = fut.result()
                completed.append(instance_id)

            now = time.monotonic()
            timed_out_futures = [
                fut
                for fut in pending
                if now - future_start_times[fut] > instance_timeout
            ]

            for fut in timed_out_futures:
                pending.discard(fut)
                timed_out.append(future_to_instance[fut])
                fut.cancel()

        assert set(completed) == {"fast_1", "fast_2"}, "Fast instances should complete"
        assert set(timed_out) == {"slow_1", "slow_2"}, "Slow instances should timeout"


@pytest.mark.asyncio
async def test_thread_pool_queue_time_not_counted_against_timeout():
    """Test that thread-pool queue time doesn't count against per-instance timeout.

    This test verifies the fix for the race condition where:
    - asyncio semaphore allows more concurrent tasks than ThreadPoolExecutor capacity
    - Tasks queue in the thread pool, but timeout clock was already ticking
    - This caused false timeouts that silently discarded completed results

    The fix moves start_time recording into the thread wrapper so only actual
    execution time counts against the timeout.
    """
    # Simulate: semaphore allows 4 concurrent, but thread pool only has 2 workers
    # This creates queue pressure where 2 tasks wait in thread pool queue
    semaphore = asyncio.Semaphore(4)
    thread_pool_workers = 2
    instance_timeout = 0.5  # 500ms timeout

    # Create pending instances tracking
    pending_instances: dict[asyncio.Task, MockPendingInstance] = {}
    completed: list[str] = []
    timed_out: list[str] = []

    # Use an explicit ThreadPoolExecutor to control worker count
    executor = ThreadPoolExecutor(max_workers=thread_pool_workers)

    async def process_with_semaphore(inst_id: str) -> tuple[str, dict]:
        """Process one instance with semaphore-based concurrency control."""
        async with semaphore:
            task = asyncio.current_task()
            pending_info = pending_instances.get(task) if task is not None else None

            def _thread_wrapper() -> tuple[str, dict]:
                # Record start time when the thread actually begins executing,
                # NOT when the semaphore was acquired. This avoids counting
                # thread-pool queue time against the per-instance timeout.
                if pending_info is not None:
                    pending_info.start_time = time.monotonic()
                # Simulate 200ms of actual work - well under timeout
                time.sleep(0.2)
                return inst_id, {"status": "completed"}

            # Use explicit executor to control worker count
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, _thread_wrapper)

    try:
        # Create 4 tasks (2x thread pool capacity)
        tasks = []
        for i in range(4):
            task = asyncio.create_task(process_with_semaphore(f"inst_{i}"))
            pending_instances[task] = MockPendingInstance(instance_id=f"inst_{i}")
            tasks.append(task)

        # Simulate queue delay by waiting before processing starts
        # In real scenario, semaphore allows all 4 tasks, but only 2 can run in threads
        # The other 2 queue in ThreadPoolExecutor with timeout clock ticking
        await asyncio.sleep(0.1)  # Simulate queue time

        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(
                pending, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                try:
                    inst_id, _ = task.result()
                    completed.append(inst_id)
                except Exception:
                    pass

            # Check for timeouts based on start_time (set in thread wrapper)
            now = time.monotonic()
            for task in list(pending):
                info = pending_instances.get(task)
                # Only check timeout if thread has started (start_time is set)
                if info and info.start_time is not None:
                    if now - info.start_time > instance_timeout:
                        timed_out.append(info.instance_id)
                        task.cancel()
                        pending.discard(task)

    finally:
        executor.shutdown(wait=False)

    # All 4 instances should complete - none should timeout
    # Each instance only does 200ms of actual work, well under 500ms timeout
    # Queue time is NOT counted because start_time is set when thread starts
    assert len(completed) == 4, (
        f"All instances should complete, but only {len(completed)} did. "
        f"Timed out: {timed_out}"
    )
    assert len(timed_out) == 0, (
        f"No instances should timeout since actual work (200ms) < timeout (500ms). "
        f"Timed out: {timed_out}"
    )


@pytest.mark.asyncio
async def test_thread_pool_queue_time_would_cause_false_timeout_without_fix():
    """Demonstrate the bug that the fix addresses.

    This test shows that if start_time were set at semaphore acquisition
    (before the fix), queue time would be counted against the timeout,
    causing false timeouts for instances that complete successfully.
    """
    # Simulate: semaphore allows 4 concurrent, but thread pool only has 1 worker
    # With only 1 worker, tasks must execute sequentially, maximizing queue time
    semaphore = asyncio.Semaphore(4)
    thread_pool_workers = 1
    instance_timeout = 0.3  # Short timeout to trigger false positives

    pending_instances: dict[asyncio.Task, MockPendingInstance] = {}
    completed: list[str] = []
    timed_out: list[str] = []

    # Use an explicit ThreadPoolExecutor to control worker count
    executor = ThreadPoolExecutor(max_workers=thread_pool_workers)

    async def process_with_old_behavior(inst_id: str) -> tuple[str, dict]:
        """Process with the OLD (buggy) behavior: start_time at semaphore acquisition."""
        async with semaphore:
            task = asyncio.current_task()
            pending_info = pending_instances.get(task) if task is not None else None

            # OLD BEHAVIOR (BUG): Set start_time BEFORE thread starts
            # This counts queue time against the timeout
            if pending_info is not None:
                pending_info.start_time = time.monotonic()

            def _thread_wrapper() -> tuple[str, dict]:
                # Simulate 100ms of actual work
                time.sleep(0.1)
                return inst_id, {"status": "completed"}

            # Use explicit executor to control worker count
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, _thread_wrapper)

    try:
        # Create 4 tasks, but only 1 worker means they must run sequentially
        # Task 1: runs immediately (0-100ms)
        # Task 2: queued ~100ms, runs 100-200ms (total wait: 100ms before work starts)
        # Task 3: queued ~200ms, runs 200-300ms (total wait: 200ms before work starts)
        # Task 4: queued ~300ms, runs 300-400ms (total wait: 300ms before work starts)
        # With OLD behavior, tasks 3 and 4 would timeout (>300ms from semaphore acquire)
        tasks = []
        for i in range(4):
            task = asyncio.create_task(process_with_old_behavior(f"inst_{i}"))
            pending_instances[task] = MockPendingInstance(instance_id=f"inst_{i}")
            tasks.append(task)

        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(
                pending, timeout=0.05, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                try:
                    inst_id, _ = task.result()
                    completed.append(inst_id)
                except Exception:
                    pass

            # Check for timeouts (same logic, but start_time was set too early)
            now = time.monotonic()
            for task in list(pending):
                info = pending_instances.get(task)
                if info and info.start_time is not None:
                    if now - info.start_time > instance_timeout:
                        timed_out.append(info.instance_id)
                        task.cancel()
                        pending.discard(task)

    finally:
        executor.shutdown(wait=False)

    # With OLD behavior, later tasks timeout due to queue time being counted
    # This demonstrates the bug the fix addresses
    assert len(timed_out) > 0, (
        "With OLD behavior (start_time at semaphore), queue time is counted "
        "against timeout, causing false timeouts"
    )
    assert len(completed) < 4, (
        "With OLD behavior, not all instances complete due to false timeouts"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
