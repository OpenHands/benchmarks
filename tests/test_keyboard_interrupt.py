"""Tests for KeyboardInterrupt handling in the evaluation module."""

import os
import signal
import subprocess
import sys
import tempfile
import time

import psutil
import pytest


# Helper script that will be run as subprocess
EVALUATION_SCRIPT = """
import os
import time
import sys
from typing import List
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, "{project_root}")

from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import LLM
from openhands.sdk.critic import PassCritic
from openhands.sdk.workspace import RemoteWorkspace


class TestEvaluation(Evaluation):
    def prepare_instances(self) -> List[EvalInstance]:
        return [
            EvalInstance(id=f"test_instance_{{i}}", data={{"test": "data"}})
            for i in range(10)
        ]

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        mock_workspace = Mock(spec=RemoteWorkspace)
        mock_workspace.__enter__ = Mock(return_value=mock_workspace)
        mock_workspace.__exit__ = Mock(return_value=None)
        mock_workspace.forward_env = forward_env or []
        mock_workspace.resource_factor = resource_factor
        return mock_workspace

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        # Signal that this worker has started
        open(os.path.join("{tmpdir}", f"worker_started_{{instance.id}}"), "w").close()
        # Simulate long-running task
        time.sleep(60)  # Long sleep
        return EvalOutput(
            instance_id=instance.id,
            test_result={{"success": True}},
            instruction="test instruction",
            error=None,
            history=[],
            instance=instance.data,
        )


if __name__ == "__main__":
    llm = LLM(model="test-model")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir="{tmpdir}",
        details={{}},
        eval_limit=0,
        n_critic_runs=1,
        max_retries=0,
        critic=PassCritic(),
    )

    evaluation = TestEvaluation(metadata=metadata, num_workers=4)

    print("PID:{{}}".format(os.getpid()), flush=True)

    try:
        evaluation.run()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught", flush=True)
        sys.exit(0)
"""


def get_child_processes(parent_pid: int) -> list:
    """Get all child processes of a parent process recursively."""
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        return children
    except psutil.NoSuchProcess:
        return []


def test_keyboard_interrupt_cleanup():
    """Test that worker threads are properly cleaned up on KeyboardInterrupt.

    The asyncio evaluation uses threads (via asyncio.to_thread()), not child
    processes. This test verifies that:
    1. Worker threads are running before the interrupt
    2. The process exits cleanly after SIGINT (which implies all threads stop)
    """
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the test script
        script_path = os.path.join(tmpdir, "test_eval.py")
        with open(script_path, "w") as f:
            f.write(EVALUATION_SCRIPT.format(project_root=project_root, tmpdir=tmpdir))

        # Start the evaluation in a subprocess
        print("\n=== Starting evaluation subprocess ===")
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Wait for the process to start and get its PID
        eval_pid = None
        start_time = time.time()
        stdout_lines = []

        assert process.stdout is not None, "Process stdout is None"

        while time.time() - start_time < 10:
            # Check if process is still running
            if process.poll() is not None:
                # Process died, get all output
                stdout_rest, stderr_rest = process.communicate()
                print(f"Process died with code: {process.returncode}")
                print(f"STDOUT: {stdout_rest}")
                print(f"STDERR: {stderr_rest}")
                break

            try:
                # Try to read the PID from stdout
                line = process.stdout.readline()
                if line:
                    print(f"Got line: {line.strip()}")
                    stdout_lines.append(line)
                    if line.startswith("PID:"):
                        eval_pid = int(line.split(":")[1].strip())
                        print(f"Evaluation process PID: {eval_pid}")
                        break
            except Exception as e:
                print(f"Error reading PID: {e}")
            time.sleep(0.1)

        if eval_pid is None and process.stderr is not None:
            # Try to get any error output
            try:
                stderr_content = process.stderr.read()
                print(f"\nSTDERR output:\n{stderr_content}")
            except Exception:
                pass

        assert eval_pid is not None, (
            f"Could not get evaluation process PID. Stdout: {stdout_lines}"
        )

        # Wait for at least one worker thread to start by polling for
        # sentinel files written by evaluate_instance().
        print("Waiting for workers to start...")
        for _ in range(100):  # 10 seconds max
            started = [f for f in os.listdir(tmpdir) if f.startswith("worker_started_")]
            if started:
                print(f"Workers started: {len(started)} sentinel(s) found")
                break
            assert process.poll() is None, (
                f"Process exited prematurely with code {process.returncode}"
            )
            time.sleep(0.1)
        else:
            pytest.fail("Workers never started (no sentinel files found)")

        # Send SIGINT to the subprocess
        print("\n=== Sending SIGINT ===")
        process.send_signal(signal.SIGINT)

        # Wait for process to exit — clean exit proves all threads stopped
        try:
            process.wait(timeout=10)
            print(f"Process exited with code: {process.returncode}")
        except subprocess.TimeoutExpired:
            print("Process did not exit in time, force killing")
            process.kill()
            process.wait()

        # Verify the process is fully gone (no zombie / leftover children)
        time.sleep(1)
        remaining_children = get_child_processes(eval_pid)
        assert len(remaining_children) == 0, (
            f"Child processes still running after SIGINT: "
            f"{[(p.pid, p.name()) for p in remaining_children]}"
        )

        print("✓ Process exited cleanly — all worker threads cleaned up")


def test_keyboard_interrupt_immediate():
    """Test cleanup when interrupt happens very early."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "test_eval.py")
        with open(script_path, "w") as f:
            f.write(EVALUATION_SCRIPT.format(project_root=project_root, tmpdir=tmpdir))

        print("\n=== Testing immediate interrupt ===")
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Get PID
        eval_pid = None
        assert process.stdout is not None, "Process stdout is None"
        for _ in range(50):
            try:
                line = process.stdout.readline()
                if line.startswith("PID:"):
                    eval_pid = int(line.split(":")[1].strip())
                    break
            except Exception:
                pass
            time.sleep(0.1)

        assert eval_pid is not None, "Could not get PID"

        # Send interrupt almost immediately
        time.sleep(0.5)
        process.send_signal(signal.SIGINT)

        # Wait for cleanup
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

        time.sleep(1)

        # Verify no zombie processes
        try:
            parent = psutil.Process(eval_pid)
            remaining = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            remaining = []

        python_workers = [p for p in remaining if "python" in p.name().lower()]

        assert len(python_workers) == 0, (
            f"Worker processes still running: {[p.pid for p in python_workers]}"
        )

        print("✓ Immediate interrupt handled correctly")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
