"""Tests for iterative evaluation resume functionality."""

import json
import os
import tempfile
from typing import List
from unittest.mock import Mock

from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import LLM
from openhands.sdk.critic import CriticResult, PassCritic
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

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """Return a mock workspace."""
        mock_workspace = Mock(spec=RemoteWorkspace)
        mock_workspace.__enter__ = Mock(return_value=mock_workspace)
        mock_workspace.__exit__ = Mock(return_value=None)
        mock_workspace.forward_env = forward_env or []
        mock_workspace.resource_factor = resource_factor
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
            history=[],  # Empty history for testing
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
                        history=[],  # Empty history for testing
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
            critic=PassCritic(),
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
                        history=[],  # Empty history for testing
                        instance=inst.data,
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

        def track_on_result(instance: EvalInstance, output: EvalOutput):
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


def test_retry_includes_missing_instances_from_prev_attempt():
    """Test that instances missing from the previous attempt file are retried.

    Scenario (from PR review):
      all_instances = [A, B, C, D]
      prev attempt file (attempt 1) has: A (failed), B (success)
      current attempt file (attempt 2) has: A (success)

    Expected:
      - C, D should be retried (missing from prev attempt)
      - A should NOT be retried (already completed in current attempt)
      - B should NOT be retried (succeeded in prev attempt)
    """

    class PatchRequiredCritic(PassCritic):
        """Fails instances without a git_patch, passes those with one."""

        def evaluate(self, events, git_patch=None):
            if git_patch:
                return CriticResult(score=1.0, message="Has patch")
            return CriticResult(score=0.0, message="No patch")

    with tempfile.TemporaryDirectory() as tmpdir:
        all_instances = [
            EvalInstance(id=inst_id, data={"test": inst_id})
            for inst_id in ["A", "B", "C", "D"]
        ]

        critic = PatchRequiredCritic()
        llm = LLM(model="test-model", temperature=0.0)

        metadata = EvalMetadata(
            llm=llm,
            dataset="test",
            dataset_split="test",
            max_iterations=10,
            eval_output_dir=tmpdir,
            details={},
            eval_limit=4,
            max_attempts=3,
            max_retries=0,
            critic=critic,
        )

        evaluation = MockEvaluation(
            metadata=metadata,
            num_workers=1,
            instances=all_instances,
        )

        # Write prev attempt file (attempt 1): A (failed), B (success)
        # C and D are intentionally absent (simulating crash / no output)
        prev_file = os.path.join(tmpdir, "output.critic_attempt_1.jsonl")
        with open(prev_file, "w") as f:
            # A: failed (no git_patch → critic returns score=0)
            output_a = EvalOutput(
                instance_id="A",
                test_result={},
                instruction="mock",
                error=None,
                history=[],
                instance={"test": "A"},
            )
            f.write(output_a.model_dump_json() + "\n")
            # B: success (has git_patch → critic returns score=1)
            output_b = EvalOutput(
                instance_id="B",
                test_result={"git_patch": "valid patch"},
                instruction="mock",
                error=None,
                history=[],
                instance={"test": "B"},
            )
            f.write(output_b.model_dump_json() + "\n")

        # Write current attempt file (attempt 2): A already completed
        current_file = os.path.join(tmpdir, "output.critic_attempt_2.jsonl")
        with open(current_file, "w") as f:
            output_a_current = EvalOutput(
                instance_id="A",
                test_result={"git_patch": "fixed patch"},
                instruction="mock",
                error=None,
                history=[],
                instance={"test": "A"},
            )
            f.write(output_a_current.model_dump_json() + "\n")

        # Call _get_instances_for_attempt for attempt 2
        result = evaluation._get_instances_for_attempt(
            attempt=2,
            all_instances=all_instances,
            critic=critic,
        )

        result_ids = {inst.id for inst in result}

        # C and D should be retried (missing from prev attempt)
        # A should NOT be retried (already completed in current attempt)
        # B should NOT be retried (succeeded in prev, not in failed_in_prev or missing_in_prev)
        assert result_ids == {"C", "D"}, (
            f"Expected to retry C and D (missing from prev attempt), "
            f"but got: {result_ids}"
        )


def test_passed_instances_not_retried_in_later_attempts():
    """Test that instances which passed in an earlier attempt are not re-retried.

    Regression test: when computing "missing" instances, the code must check
    ALL prior attempt files, not just the immediately previous one. Otherwise
    an instance that passed in attempt 1 (and therefore has no entry in attempt
    2's file) would look "missing" and get incorrectly retried in attempt 3.

    Scenario:
      all_instances = [A, B, C, D]
      attempt 1 file: A (pass), B (fail)         — C, D crashed
      attempt 2 file: B (pass), C (pass), D (fail) — only retried B, C, D
      current = attempt 3 (empty file)

    Expected for attempt 3:
      - D should be retried  (failed in attempt 2)
      - A should NOT be retried (passed in attempt 1, absent from attempt 2 is fine)
      - B, C should NOT be retried (passed in attempt 2)
    """

    class PatchRequiredCritic(PassCritic):
        """Fails instances without a git_patch, passes those with one."""

        def evaluate(self, events, git_patch=None):
            if git_patch:
                return CriticResult(score=1.0, message="Has patch")
            return CriticResult(score=0.0, message="No patch")

    with tempfile.TemporaryDirectory() as tmpdir:
        all_instances = [
            EvalInstance(id=inst_id, data={"test": inst_id})
            for inst_id in ["A", "B", "C", "D"]
        ]

        critic = PatchRequiredCritic()
        llm = LLM(model="test-model", temperature=0.0)

        metadata = EvalMetadata(
            llm=llm,
            dataset="test",
            dataset_split="test",
            max_iterations=10,
            eval_output_dir=tmpdir,
            details={},
            eval_limit=4,
            max_attempts=3,
            max_retries=0,
            critic=critic,
        )

        evaluation = MockEvaluation(
            metadata=metadata,
            num_workers=1,
            instances=all_instances,
        )

        def _make_output(instance_id, has_patch):
            return EvalOutput(
                instance_id=instance_id,
                test_result={"git_patch": "patch"} if has_patch else {},
                instruction="mock",
                error=None,
                history=[],
                instance={"test": instance_id},
            )

        # Attempt 1 file: A passes, B fails.  C, D missing (crashed).
        attempt1_file = os.path.join(tmpdir, "output.critic_attempt_1.jsonl")
        with open(attempt1_file, "w") as f:
            f.write(_make_output("A", has_patch=True).model_dump_json() + "\n")
            f.write(_make_output("B", has_patch=False).model_dump_json() + "\n")

        # Attempt 2 file: B, C, D were retried. B passes, C passes, D fails.
        # A is correctly absent (it already passed in attempt 1).
        attempt2_file = os.path.join(tmpdir, "output.critic_attempt_2.jsonl")
        with open(attempt2_file, "w") as f:
            f.write(_make_output("B", has_patch=True).model_dump_json() + "\n")
            f.write(_make_output("C", has_patch=True).model_dump_json() + "\n")
            f.write(_make_output("D", has_patch=False).model_dump_json() + "\n")

        # Attempt 3 file does not exist yet (no work done).

        result = evaluation._get_instances_for_attempt(
            attempt=3,
            all_instances=all_instances,
            critic=critic,
        )

        result_ids = {inst.id for inst in result}

        # Only D should be retried (failed in attempt 2).
        # A must NOT be retried — it passed in attempt 1 and is simply
        # absent from the attempt 2 file because it was never re-run.
        assert result_ids == {"D"}, (
            f"Expected to retry only D (failed in attempt 2), "
            f"but got: {result_ids}"
        )
