"""Tests for iterative evaluation resume functionality."""

import json
import os
import tempfile
from typing import List
from unittest.mock import Mock

from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import LLM
from openhands.sdk.workspace import RemoteWorkspace


class MockEvaluation(Evaluation):
    """Mock evaluation class for testing."""

    def __init__(self, *args, instances: List[EvalInstance], **kwargs):
        super().__init__(*args, **kwargs)
        # Store as instance variable after Pydantic initialization
        object.__setattr__(self, "_test_instances", instances)

    def prepare_instances(self) -> List[EvalInstance]:
        """Return pre-configured instances."""
        return object.__getattribute__(self, "_test_instances")

    def prepare_workspace(self, instance: EvalInstance) -> RemoteWorkspace:
        """Return a mock workspace."""
        mock_workspace = Mock(spec=RemoteWorkspace)
        mock_workspace.__enter__ = Mock(return_value=mock_workspace)
        mock_workspace.__exit__ = Mock(return_value=None)
        return mock_workspace

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """Return a mock output."""
        return EvalOutput(
            instance_id=instance.id,
            test_result={"git_patch": "mock patch"},
            instruction="mock instruction",
            error=None,
            history=[
                {
                    "kind": "ActionEvent",
                    "action": {"kind": "FinishAction"},
                }
            ],
            instance=instance.data,
        )


def test_iterative_resume_with_expanded_n_limit():
    """
    Test that iterative evaluation correctly handles resume when n-limit is expanded.

    Scenario:
    1. First run: Process 50 instances with max_attempts=3
    2. Second run: Expand to 200 instances with max_attempts=3

    Expected behavior:
    - The 150 new instances (51-200) should be processed starting from attempt 1
    - Previously completed instances (1-50) should not be re-processed
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test instances
        # Simulate first run with 50 instances
        first_50_instances = [
            EvalInstance(id=f"instance_{i}", data={"test": f"data_{i}"})
            for i in range(1, 51)
        ]

        # Create LLM config
        llm = LLM(model="test-model", temperature=0.0)

        # Simulate first run by creating output files
        # Create outputs for all 3 attempts with all 50 instances
        for attempt in range(1, 4):
            attempt_file = os.path.join(
                tmpdir, f"output.critic_attempt_{attempt}.jsonl"
            )
            with open(attempt_file, "w") as f:
                for inst in first_50_instances:
                    output = EvalOutput(
                        instance_id=inst.id,
                        test_result={"git_patch": "mock patch"},
                        instruction="mock instruction",
                        error=None,
                        history=[
                            {
                                "kind": "ActionEvent",
                                "action": {"kind": "FinishAction"},
                            }
                        ],
                        instance=inst.data,
                    )
                    f.write(output.model_dump_json() + "\n")

        # Now simulate second run with 200 instances
        all_200_instances = [
            EvalInstance(id=f"instance_{i}", data={"test": f"data_{i}"})
            for i in range(1, 201)
        ]

        # Create metadata for second run (expanded n-limit)
        metadata_run2 = EvalMetadata(
            llm=llm,
            dataset="test",
            dataset_split="test",
            max_iterations=10,
            eval_output_dir=tmpdir,
            details={},
            eval_limit=200,
            max_attempts=3,
            max_retries=0,
            critic_name="pass",
        )

        # Create evaluation with expanded instances
        evaluation = MockEvaluation(
            metadata=metadata_run2,
            num_workers=1,
            instances=all_200_instances,
        )

        # Check resume logic
        start_attempt, previous_outputs = evaluation._get_resume_start_attempt()

        # Should resume from attempt 3 (last completed attempt)
        assert start_attempt == 3, f"Expected start_attempt=3, got {start_attempt}"

        # Should have loaded 150 outputs (50 instances * 3 attempts)
        assert len(previous_outputs) == 150, (
            f"Expected 150 previous outputs (50 instances * 3 attempts), "
            f"got {len(previous_outputs)}"
        )

        # Now test the critical part: _run_iterative_mode should process new instances
        # We'll track what instances are actually processed
        processed_instances = set()

        def track_on_result(instance: EvalInstance, output: EvalOutput):
            processed_instances.add(instance.id)

        # Run evaluation (this will test the actual resume logic)
        evaluation.run(on_result=track_on_result)

        # Check that new instances were processed
        # The new instances should start from attempt 1
        expected_new_instances = {f"instance_{i}" for i in range(51, 201)}

        # Verify that at least some new instances were processed
        # (in the actual run, all should be processed, but since we're using pass critic,
        # they should all be marked as successful in attempt 1)
        new_instances_processed = processed_instances & expected_new_instances

        assert len(new_instances_processed) > 0, (
            f"Expected new instances (51-200) to be processed, "
            f"but got: {processed_instances}"
        )

        # Verify that old instances (1-50) were NOT re-processed
        old_instances = {f"instance_{i}" for i in range(1, 51)}
        old_instances_reprocessed = processed_instances & old_instances

        assert len(old_instances_reprocessed) == 0, (
            f"Old instances (1-50) should not be re-processed, "
            f"but found: {old_instances_reprocessed}"
        )

        # Check that attempt 1 file now includes the new instances
        attempt_1_file = os.path.join(tmpdir, "output.critic_attempt_1.jsonl")
        with open(attempt_1_file, "r") as f:
            attempt_1_instances = set()
            for line in f:
                output = EvalOutput(**json.loads(line))
                attempt_1_instances.add(output.instance_id)

        # Attempt 1 should have all 200 instances now
        # (50 from first run + 150 new from second run)
        assert len(attempt_1_instances) == 200, (
            f"Expected 200 instances in attempt 1 file, got {len(attempt_1_instances)}"
        )


def test_iterative_resume_with_same_n_limit():
    """
    Test that resume works correctly when n-limit stays the same.

    This is the normal resume case - should continue where it left off.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 50 test instances
        instances = [
            EvalInstance(id=f"instance_{i}", data={"test": f"data_{i}"})
            for i in range(1, 51)
        ]

        llm = LLM(model="test-model", temperature=0.0)

        metadata = EvalMetadata(
            llm=llm,
            dataset="test",
            dataset_split="test",
            max_iterations=10,
            eval_output_dir=tmpdir,
            details={},
            eval_limit=50,
            max_attempts=3,
            max_retries=0,
            critic_name="pass",
        )

        # Simulate partial run - only attempt 1 and 2 completed
        for attempt in range(1, 3):
            attempt_file = os.path.join(
                tmpdir, f"output.critic_attempt_{attempt}.jsonl"
            )
            with open(attempt_file, "w") as f:
                for inst in instances:
                    output = EvalOutput(
                        instance_id=inst.id,
                        test_result={"git_patch": "mock patch"},
                        instruction="mock instruction",
                        error=None,
                        history=[
                            {
                                "kind": "ActionEvent",
                                "action": {"kind": "FinishAction"},
                            }
                        ],
                        instance=inst.data,
                    )
                    f.write(output.model_dump_json() + "\n")

        # Create evaluation
        evaluation = MockEvaluation(
            metadata=metadata,
            num_workers=1,
            instances=instances,
        )

        # Check resume logic
        start_attempt, previous_outputs = evaluation._get_resume_start_attempt()

        # Should resume from attempt 2 (last completed attempt)
        assert start_attempt == 2, f"Expected start_attempt=2, got {start_attempt}"

        # Should have loaded 100 outputs (50 instances * 2 attempts)
        assert len(previous_outputs) == 100, (
            f"Expected 100 previous outputs, got {len(previous_outputs)}"
        )
