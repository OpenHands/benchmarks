"""Tests for per-instance timeout handling in the evaluation module."""

import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import pytest


def slow_worker(instance_id: str, sleep_time: float) -> tuple[str, dict]:
    """Simulate a slow worker that takes sleep_time seconds."""
    time.sleep(sleep_time)
    return instance_id, {"status": "completed"}


def test_per_instance_timeout_logic():
    """Test that per-instance timeout logic correctly identifies timed-out futures."""
    instance_timeout = 2  # 2 seconds for test

    with ProcessPoolExecutor(max_workers=4) as pool:
        # Submit jobs with different durations
        futures = []
        future_to_instance = {}
        future_start_times = {}

        # Fast job (should complete)
        fut1 = pool.submit(slow_worker, "fast_instance", 0.5)
        futures.append(fut1)
        future_to_instance[fut1] = "fast_instance"
        future_start_times[fut1] = time.monotonic()

        # Slow job (should timeout)
        fut2 = pool.submit(slow_worker, "slow_instance", 10.0)
        futures.append(fut2)
        future_to_instance[fut2] = "slow_instance"
        future_start_times[fut2] = time.monotonic()

        pending = set(futures)
        completed = []
        timed_out = []

        while pending:
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)

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
    instance_timeout = 5  # 5 seconds

    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = []
        future_to_instance = {}
        future_start_times = {}

        for i in range(3):
            fut = pool.submit(slow_worker, f"instance_{i}", 0.1)
            futures.append(fut)
            future_to_instance[fut] = f"instance_{i}"
            future_start_times[fut] = time.monotonic()

        pending = set(futures)
        completed = []
        timed_out = []

        while pending:
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)

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
    instance_timeout = 1  # 1 second

    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = []
        future_to_instance = {}
        future_start_times = {}

        # Mix of fast and slow jobs
        configs = [
            ("fast_1", 0.1),
            ("slow_1", 10.0),
            ("fast_2", 0.2),
            ("slow_2", 10.0),
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
            done, pending = wait(pending, timeout=0.3, return_when=FIRST_COMPLETED)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
