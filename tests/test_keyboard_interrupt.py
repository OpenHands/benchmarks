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
        self, instance: EvalInstance, resource_factor: int = 1
    ) -> RemoteWorkspace:
        mock_workspace = Mock(spec=RemoteWorkspace)
        mock_workspace.__enter__ = Mock(return_value=mock_workspace)
        mock_workspace.__exit__ = Mock(return_value=None)
        return mock_workspace

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
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
        max_attempts=1,
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
    """Test that all child processes are properly cleaned up on KeyboardInterrupt."""
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

        # Wait for worker processes to start
        print("Waiting for workers to start...")
        time.sleep(3)

        # Get child processes before interrupt
        children_before = get_child_processes(eval_pid)
        python_workers_before = [
            p for p in children_before if "python" in p.name().lower()
        ]
        print(f"Worker processes before interrupt: {len(python_workers_before)}")
        print(f"Worker PIDs: {[p.pid for p in python_workers_before]}")

        # Verify we have worker processes
        assert len(python_workers_before) > 0, (
            f"No worker processes found. All children: {[(p.pid, p.name()) for p in children_before]}"
        )

        # Send SIGINT to the subprocess
        print("\n=== Sending SIGINT ===")
        process.send_signal(signal.SIGINT)

        # Wait for process to exit
        try:
            process.wait(timeout=10)
            print(f"Process exited with code: {process.returncode}")
        except subprocess.TimeoutExpired:
            print("Process did not exit in time, force killing")
            process.kill()
            process.wait()

        # Give a moment for cleanup
        time.sleep(2)

        # Check if all worker processes are gone
        remaining_workers = []
        for worker in python_workers_before:
            try:
                if psutil.pid_exists(worker.pid):
                    proc = psutil.Process(worker.pid)
                    # Check if it's still the same process (not reused PID)
                    if proc.create_time() == worker.create_time():
                        remaining_workers.append(worker.pid)
            except psutil.NoSuchProcess:
                pass  # Process is gone, which is what we want

        print("\n=== Results ===")
        print(f"Worker processes before: {len(python_workers_before)}")
        print(f"Remaining workers: {len(remaining_workers)}")
        if remaining_workers:
            print(f"Remaining PIDs: {remaining_workers}")

        # Assert all workers are cleaned up
        assert len(remaining_workers) == 0, (
            f"Worker processes still running after SIGINT: {remaining_workers}"
        )

        print("✓ All child processes cleaned up successfully")


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
