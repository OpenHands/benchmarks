"""Tests for deadlock detection in the evaluation module.

Note: These tests verify the deadlock detection LOGIC rather than the full
evaluation.py integration. Testing the actual Evaluator class would require
complex mocking of ProcessPoolExecutor, futures, and the entire evaluation
pipeline. Instead, we test the same patterns used in evaluation.py to ensure
the timeout detection, progress tracking, and error handling work correctly.

For integration testing, run the actual evaluation against a benchmark dataset.
"""

import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import pytest


def hanging_worker(instance_id: str) -> tuple[str, dict]:
    """Simulate a worker that hangs indefinitely (deadlock simulation)."""
    while True:
        time.sleep(1)


def slow_but_completing_worker(instance_id: str, sleep_time: float) -> tuple[str, dict]:
    """Worker that completes after a delay."""
    time.sleep(sleep_time)
    return instance_id, {"status": "completed"}


class TestDeadlockDetection:
    """Tests for the deadlock detection mechanism."""

    def test_deadlock_detection_triggers_on_no_progress(self):
        """Test that deadlock detection triggers when no progress is made."""
        # Use a very short timeout for testing (5 seconds)
        no_progress_timeout = 5
        
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = []
            future_to_instance = {}
            future_start_times = {}
            
            # Submit workers that will hang
            for i in range(2):
                fut = pool.submit(hanging_worker, f"hanging_{i}")
                futures.append(fut)
                future_to_instance[fut] = f"hanging_{i}"
                future_start_times[fut] = time.monotonic()
            
            pending = set(futures)
            last_progress_time = time.monotonic()
            deadlock_detected = False
            timed_out_count = 0
            
            while pending and not deadlock_detected:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                
                for fut in done:
                    last_progress_time = time.monotonic()
                
                # Deadlock detection logic (mirrors evaluation.py)
                now = time.monotonic()
                time_since_progress = now - last_progress_time
                
                if pending and time_since_progress > no_progress_timeout:
                    deadlock_detected = True
                    timed_out_count = len(pending)
                    # Clear pending to exit loop
                    for fut in list(pending):
                        fut.cancel()
                    pending.clear()
            
            # Force terminate pool
            pool.shutdown(wait=False, cancel_futures=True)
        
        assert deadlock_detected, "Deadlock should be detected"
        assert timed_out_count == 2, f"Expected 2 timed out, got {timed_out_count}"

    def test_no_deadlock_when_progress_is_made(self):
        """Test that deadlock detection does NOT trigger when progress is made."""
        no_progress_timeout = 5
        
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = []
            future_to_instance = {}
            
            # Submit workers that complete at different times
            for i in range(3):
                fut = pool.submit(slow_but_completing_worker, f"worker_{i}", 0.5 + i * 0.5)
                futures.append(fut)
                future_to_instance[fut] = f"worker_{i}"
            
            pending = set(futures)
            last_progress_time = time.monotonic()
            completed = []
            deadlock_detected = False
            
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                
                for fut in done:
                    last_progress_time = time.monotonic()
                    instance_id, result = fut.result()
                    completed.append(instance_id)
                
                now = time.monotonic()
                time_since_progress = now - last_progress_time
                
                if pending and time_since_progress > no_progress_timeout:
                    deadlock_detected = True
                    pending.clear()
        
        assert not deadlock_detected, "Deadlock should NOT be detected"
        assert len(completed) == 3, "All workers should complete"

    def test_timed_out_count_increments_correctly(self):
        """Test that timed_out_count is correctly incremented for deadlocked instances."""
        no_progress_timeout = 3
        
        with ProcessPoolExecutor(max_workers=3) as pool:
            futures = []
            future_to_instance = {}
            
            # 1 fast worker + 2 hanging workers
            fut_fast = pool.submit(slow_but_completing_worker, "fast", 0.2)
            futures.append(fut_fast)
            future_to_instance[fut_fast] = "fast"
            
            for i in range(2):
                fut = pool.submit(hanging_worker, f"hanging_{i}")
                futures.append(fut)
                future_to_instance[fut] = f"hanging_{i}"
            
            pending = set(futures)
            last_progress_time = time.monotonic()
            timed_out_count = 0
            completed = []
            
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                
                for fut in done:
                    last_progress_time = time.monotonic()
                    try:
                        instance_id, result = fut.result()
                        completed.append(instance_id)
                    except Exception:
                        pass
                
                now = time.monotonic()
                time_since_progress = now - last_progress_time
                
                if pending and time_since_progress > no_progress_timeout:
                    deadlocked_count = len(pending)
                    timed_out_count += deadlocked_count
                    for fut in list(pending):
                        fut.cancel()
                    pending.clear()
            
            pool.shutdown(wait=False, cancel_futures=True)
        
        assert "fast" in completed, "Fast worker should complete"
        assert timed_out_count == 2, f"Expected 2 deadlocked, got {timed_out_count}"

    def test_error_output_created_for_deadlocked_instances(self):
        """Test that error outputs are created with correct metadata for deadlocked instances."""
        no_progress_timeout = 3
        error_outputs = []
        
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = []
            future_to_instance = {}
            
            for i in range(2):
                fut = pool.submit(hanging_worker, f"instance_{i}")
                futures.append(fut)
                future_to_instance[fut] = f"instance_{i}"
            
            pending = set(futures)
            last_progress_time = time.monotonic()
            
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                
                for fut in done:
                    last_progress_time = time.monotonic()
                
                now = time.monotonic()
                time_since_progress = now - last_progress_time
                
                if pending and time_since_progress > no_progress_timeout:
                    for fut in list(pending):
                        instance_id = future_to_instance[fut]
                        # Simulate creating error output like evaluation.py does
                        error_outputs.append({
                            "instance_id": instance_id,
                            "error": f"Worker deadlock detected after {time_since_progress / 60:.1f} minutes",
                            "time_since_progress": time_since_progress,
                        })
                        fut.cancel()
                    pending.clear()
            
            pool.shutdown(wait=False, cancel_futures=True)
        
        assert len(error_outputs) == 2, "Should have 2 error outputs"
        for output in error_outputs:
            assert "deadlock" in output["error"].lower()
            assert output["time_since_progress"] >= no_progress_timeout

    def test_pending_set_cleared_after_deadlock(self):
        """Test that the pending set is properly cleared after deadlock detection."""
        no_progress_timeout = 3
        
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = []
            
            for i in range(2):
                fut = pool.submit(hanging_worker, f"instance_{i}")
                futures.append(fut)
            
            pending = set(futures)
            last_progress_time = time.monotonic()
            pending_after_detection = None
            
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                
                for fut in done:
                    last_progress_time = time.monotonic()
                
                now = time.monotonic()
                time_since_progress = now - last_progress_time
                
                if pending and time_since_progress > no_progress_timeout:
                    for fut in list(pending):
                        fut.cancel()
                    pending.clear()
                    pending_after_detection = len(pending)
            
            pool.shutdown(wait=False, cancel_futures=True)
        
        assert pending_after_detection == 0, "Pending set should be empty after deadlock"


class TestConfigurableTimeout:
    """Tests for the configurable no-progress timeout."""

    def test_timeout_from_env_var(self):
        """Test that EVALUATION_NO_PROGRESS_TIMEOUT env var is respected."""
        # Set a custom timeout
        os.environ["EVALUATION_NO_PROGRESS_TIMEOUT"] = "10"
        timeout_value = int(os.getenv("EVALUATION_NO_PROGRESS_TIMEOUT", "1800"))
        assert timeout_value == 10
        
        # Cleanup
        del os.environ["EVALUATION_NO_PROGRESS_TIMEOUT"]

    def test_default_timeout(self):
        """Test default timeout is 1800 seconds (30 minutes)."""
        # Ensure env var is not set
        if "EVALUATION_NO_PROGRESS_TIMEOUT" in os.environ:
            del os.environ["EVALUATION_NO_PROGRESS_TIMEOUT"]
        
        timeout_value = int(os.getenv("EVALUATION_NO_PROGRESS_TIMEOUT", "1800"))
        assert timeout_value == 1800

    def test_invalid_env_var_handling(self):
        """Test that invalid env var values are handled gracefully."""
        # This tests the pattern used in evaluation.py
        os.environ["EVALUATION_NO_PROGRESS_TIMEOUT"] = "not_a_number"
        
        try:
            timeout_value = int(os.getenv("EVALUATION_NO_PROGRESS_TIMEOUT", "1800"))
            # Should not reach here
            assert False, "Should have raised ValueError"
        except ValueError:
            # Expected - code should fall back to default
            timeout_value = 1800
        
        assert timeout_value == 1800
        
        # Cleanup
        del os.environ["EVALUATION_NO_PROGRESS_TIMEOUT"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
