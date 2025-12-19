"""Tests for iterative evaluation resume functionality."""

import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import Mock

from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from benchmarks.utils.output_schema import CostBreakdown, StandardizedOutput
from openhands.sdk import LLM
from openhands.sdk.critic import PassCritic
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
        self, instance: EvalInstance, workspace: RemoteWorkspace, attempt: int
    ) -> EvalOutput:
        """Return a mock output."""
        return EvalOutput(
            instance_id=instance.id,
            test_result={"git_patch": "mock patch"},
            instruction="mock instruction",
            error=None,
            history=[],  # Empty history for testing
            instance=instance.data,
            status="success",
            resolved=True,
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
                    output = StandardizedOutput(
                        instance_id=inst.id,
                        attempt=attempt,
                        max_attempts=3,
                        status="success",
                        resolved=True,
                        test_result={"git_patch": "mock patch"},
                        cost=CostBreakdown(),
                        artifacts_url=f"artifacts/{inst.id}/attempt_{attempt}",
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
            critic=PassCritic(),
        )

        # Create evaluation with expanded instances
        evaluation = MockEvaluation(
            metadata=metadata_run2,
            num_workers=1,
            instances=all_200_instances,
        )

        # Track what instances are actually processed
        processed_instances = set()

        def track_on_result(
            instance: EvalInstance,
            output: EvalOutput,
            attempt: int,
            artifacts_dir: Path,
        ):
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
        attempt_1_file = Path(tmpdir) / "output.critic_attempt_1.jsonl"
        attempt_1_instances = {
            StandardizedOutput.model_validate_json(line).instance_id
            for line in attempt_1_file.read_text().splitlines()
            if line.strip()
        }

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
            critic=PassCritic(),
        )

        # Simulate partial run - only attempt 1 and 2 completed
        for attempt in range(1, 3):
            attempt_file = os.path.join(
                tmpdir, f"output.critic_attempt_{attempt}.jsonl"
            )
            with open(attempt_file, "w") as f:
                for inst in instances:
                    output = StandardizedOutput(
                        instance_id=inst.id,
                        attempt=attempt,
                        max_attempts=3,
                        status="success",
                        resolved=True,
                        test_result={"git_patch": "mock patch"},
                        cost=CostBreakdown(),
                        artifacts_url=f"artifacts/{inst.id}/attempt_{attempt}",
                    )
                    f.write(output.model_dump_json() + "\n")

        # Create evaluation
        evaluation = MockEvaluation(
            metadata=metadata,
            num_workers=1,
            instances=instances,
        )

        # Track what instances are processed
        processed_instances = set()

        def track_on_result(
            instance: EvalInstance,
            output: EvalOutput,
            attempt: int,
            artifacts_dir: Path,
        ):
            processed_instances.add(instance.id)

        # Run evaluation - should only process attempt 3 since 1 and 2 are complete
        evaluation.run(on_result=track_on_result)

        # All instances should be processed in attempt 3 (since pass critic marks all as success)
        # Actually, with pass critic, everything succeeds in attempt 1, so attempts 2 and 3 have nothing to do
        # But since we already have complete results for attempts 1 and 2, attempt 3 should run but find nothing to do

        # With the new design, all attempts iterate, but skip completed work
        # So no instances should be re-processed
        assert len(processed_instances) == 0, (
            f"Expected no instances to be re-processed (all already complete in attempts 1-2), "
            f"but found: {processed_instances}"
        )
