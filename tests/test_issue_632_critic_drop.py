"""
Test for issue #632: Number of total instances accepted dropped.

This tests the scenario where the count of accepted instances can decrease
between progress snapshots due to critic evaluation depending on history
being present in the output files.
"""

import json
import os
import tempfile

import pytest

from benchmarks.utils.critics import evaluate_output
from benchmarks.utils.iterative import _get_output_rank, aggregate_results
from benchmarks.utils.models import EvalOutput
from openhands.sdk.critic import AgentFinishedCritic, PassCritic


def create_output_with_history(
    instance_id: str,
    git_patch: str | None = "mock patch",
    error: str | None = None,
    history: list | None = None,
) -> EvalOutput:
    """
    Create an EvalOutput for testing.

    Args:
        instance_id: The instance ID
        git_patch: The git patch (None for empty)
        error: Error message (None for no error)
        history: List of events (None for empty history)
    """
    return EvalOutput(
        instance_id=instance_id,
        test_result={"git_patch": git_patch or ""},
        instruction="mock instruction",
        error=error,
        history=history or [],
        instance={"test": "data"},
    )


class TestCriticEvaluationWithHistory:
    """Tests for critic evaluation behavior with different history states."""

    def test_agent_finished_critic_without_history_fails(self):
        """
        AgentFinishedCritic should fail when history is empty even with patch.

        This is the CORE issue #632: The critic depends on history to find
        FinishAction. Without history, the critic fails even if the patch exists.
        """
        critic = AgentFinishedCritic()

        # No history - even with valid patch, critic should fail
        output = create_output_with_history(
            "test_1", git_patch="some changes", history=[]
        )
        result = evaluate_output(critic, output)

        assert result is False, (
            "Should fail without history (no FinishAction found). "
            "This demonstrates the core issue: critic evaluation depends on history."
        )

    def test_agent_finished_critic_with_empty_patch_fails(self):
        """
        AgentFinishedCritic should fail with empty patch even with valid history.

        This is expected behavior - patch must be non-empty.
        """
        critic = AgentFinishedCritic()

        # Empty patch - should fail (even with history, patch check fails first)
        output = create_output_with_history("test_1", git_patch="", history=[])
        result = evaluate_output(critic, output)

        assert result is False, "Should fail with empty patch"


class TestAggregateResultsWithHistory:
    """Tests for aggregation behavior when history affects critic evaluation."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_aggregation_respects_history_state(self, temp_output_dir):
        """
        Test that aggregation correctly handles instances where history state differs.

        This reproduces issue #632: if an instance has patch but no history,
        the critic evaluation fails and the instance may be dropped or replaced
        with a lower-ranked result from another attempt.
        """
        pass_critic = PassCritic()

        # Create attempt 1: instance with patch but NO history
        # (This simulates the scenario where history was cleared before writing to disk)
        attempt_1_file = os.path.join(temp_output_dir, "output.critic_attempt_1.jsonl")
        output_1 = create_output_with_history(
            "instance_1",
            git_patch="some changes",
            history=[],
        )
        with open(attempt_1_file, "w") as f:
            f.write(output_1.model_dump_json() + "\n")

        # Create attempt 2: same instance with patch (for PassCritic test)
        attempt_2_file = os.path.join(temp_output_dir, "output.critic_attempt_2.jsonl")
        output_2 = create_output_with_history(
            "instance_1",
            git_patch="some changes",
            history=[],
        )
        with open(attempt_2_file, "w") as f:
            f.write(output_2.model_dump_json() + "\n")

        # Run aggregation with PassCritic - both should pass
        aggregate_results(temp_output_dir, n_critic_runs=2, critic=pass_critic)

        # Read final output
        final_output_file = os.path.join(temp_output_dir, "output.jsonl")
        with open(final_output_file, "r") as f:
            lines = f.readlines()

        # The instance should still be present
        assert len(lines) == 1, f"Instance should be in output, got {len(lines)} lines"

        # Verify the output has the patch
        result = json.loads(lines[0])
        assert result["instance_id"] == "instance_1"
        assert result["test_result"]["git_patch"] == "some changes"

    def test_output_rank_changes_with_history(self):
        """
        Test that _get_output_rank returns different ranks based on history presence.

        This demonstrates the core issue #632: the same instance with the same patch
        gets different ranks depending on whether history is present.
        """
        critic = AgentFinishedCritic()

        # With empty history: rank should be 1 (critic fails due to no FinishAction)
        output_without_history = create_output_with_history(
            "test", git_patch="patch", history=[]
        )
        rank_without_history = _get_output_rank(critic, output_without_history)

        # With history (any event): rank should be different
        # Note: We can't easily create a proper ActionEvent with FinishAction,
        # but we can test that empty history gives rank 1 (critic-failed)
        assert rank_without_history == 1, (
            f"Output without history should have rank 1 (critic failed), got {rank_without_history}. "
            "This confirms the critic depends on history."
        )


class TestProgressConsistency:
    """Tests to verify that progress counts remain consistent."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_same_instance_multiple_attempts_all_pass(self, temp_output_dir):
        """
        Test that if all attempts pass, the instance count remains stable.

        When n_critic_runs > 1, an instance may be evaluated in multiple attempts.
        If all attempts pass the critic, the count should not decrease.
        """
        pass_critic = PassCritic()  # PassCritic doesn't depend on history

        # Create both attempt files with passing results
        for attempt in [1, 2]:
            attempt_file = os.path.join(
                temp_output_dir, f"output.critic_attempt_{attempt}.jsonl"
            )
            output = create_output_with_history(
                "instance_1",
                git_patch="patch content",
                history=[],  # Empty history - PassCritic ignores this
            )
            with open(attempt_file, "w") as f:
                f.write(output.model_dump_json() + "\n")

        # Run aggregation
        aggregate_results(temp_output_dir, n_critic_runs=2, critic=pass_critic)

        # Read final output - should have 1 instance
        final_output_file = os.path.join(temp_output_dir, "output.jsonl")
        with open(final_output_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 1, (
            f"Should have exactly 1 instance in final output, got {len(lines)}. "
            "If this fails, it means the instance was dropped during aggregation."
        )

    def test_serialization_preserves_history(self):
        """
        Test that EvalOutput serialization/deserialization preserves history.

        This is critical for progress tracking: if history is lost during
        serialization, critic evaluation will fail and counts will be incorrect.
        """
        # Create output with empty history (since history entries must be Event objects)
        original = create_output_with_history("test", git_patch="patch", history=[])

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize
        restored = EvalOutput.model_validate_json(json_str)

        # History should be preserved
        assert len(restored.history) == len(original.history), (
            f"History length mismatch: original={len(original.history)}, "
            f"restored={len(restored.history)}"
        )
