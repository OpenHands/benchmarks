"""Tests for aggregate_results functionality in iterative.py."""

import json
import os
import tempfile

import pytest

from benchmarks.utils.iterative import _get_output_rank, aggregate_results
from benchmarks.utils.models import EvalOutput
from openhands.sdk.critic import CriticResult, PassCritic


class FailCritic(PassCritic):
    """A critic that always fails (returns success=False)."""

    def evaluate(self, events, git_patch=None):
        return CriticResult(score=0.0, message="Always fails")


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def create_output(instance_id: str, error: str | None = None) -> EvalOutput:
    """Helper to create an EvalOutput for testing."""
    return EvalOutput(
        instance_id=instance_id,
        test_result={"git_patch": "mock patch"},
        instruction="mock instruction",
        error=error,
        history=[],
        instance={"test": "data"},
    )


class TestGetOutputRank:
    """Tests for _get_output_rank function."""

    def test_error_output_has_lowest_rank(self):
        """Error outputs should have rank 0."""
        critic = PassCritic()
        output = create_output("test_1", error="Some error")
        assert _get_output_rank(critic, output) == 0

    def test_non_error_critic_fail_has_middle_rank(self):
        """Non-error outputs that fail critic should have rank 1."""
        critic = FailCritic()
        output = create_output("test_1", error=None)
        assert _get_output_rank(critic, output) == 1

    def test_critic_success_has_highest_rank(self):
        """Critic-successful outputs should have rank 2."""
        critic = PassCritic()
        output = create_output("test_1", error=None)
        assert _get_output_rank(critic, output) == 2


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def test_prefers_non_error_over_error_when_last_attempt_errors(
        self, temp_output_dir
    ):
        """
        Test that non-error rows are preferred over error rows.

        Scenario (from issue #297):
        - Attempt 1: non-error, critic-fail
        - Attempt 2: non-error, critic-fail
        - Attempt 3: error (runtime pending/404)

        Expected: Instance should appear in output.jsonl with attempt 2's result
        (the latest non-error result).
        """
        critic = FailCritic()

        # Create attempt files
        # Attempt 1: non-error, critic-fail
        attempt_1_file = os.path.join(temp_output_dir, "output.critic_attempt_1.jsonl")
        output_1 = create_output("instance_1", error=None)
        with open(attempt_1_file, "w") as f:
            f.write(output_1.model_dump_json() + "\n")

        # Attempt 2: non-error, critic-fail
        attempt_2_file = os.path.join(temp_output_dir, "output.critic_attempt_2.jsonl")
        output_2 = create_output("instance_1", error=None)
        with open(attempt_2_file, "w") as f:
            f.write(output_2.model_dump_json() + "\n")

        # Attempt 3: error
        attempt_3_file = os.path.join(temp_output_dir, "output.critic_attempt_3.jsonl")
        output_3 = create_output("instance_1", error="Runtime pending/404")
        with open(attempt_3_file, "w") as f:
            f.write(output_3.model_dump_json() + "\n")

        # Run aggregation
        aggregate_results(temp_output_dir, max_attempts=3, critic=critic)

        # Verify output.jsonl contains the instance (not dropped)
        final_output_file = os.path.join(temp_output_dir, "output.jsonl")
        assert os.path.exists(final_output_file)

        with open(final_output_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 1, "Instance should not be dropped"
        result = json.loads(lines[0])
        assert result["instance_id"] == "instance_1"
        assert result["error"] is None

    def test_prefers_critic_success_over_non_error_critic_fail(self, temp_output_dir):
        """
        Test that critic-successful rows are preferred over non-error critic-fail rows.

        Scenario:
        - Attempt 1: non-error, critic-success
        - Attempt 2: non-error, critic-fail
        - Attempt 3: non-error, critic-fail

        Expected: Instance should use attempt 1's result (critic-successful).
        """
        critic = PassCritic()

        # Create attempt files - all non-error, all critic-success with PassCritic
        for attempt in range(1, 4):
            attempt_file = os.path.join(
                temp_output_dir, f"output.critic_attempt_{attempt}.jsonl"
            )
            output = create_output("instance_1", error=None)
            with open(attempt_file, "w") as f:
                f.write(output.model_dump_json() + "\n")

        # Run aggregation
        aggregate_results(temp_output_dir, max_attempts=3, critic=critic)

        # Verify output.jsonl contains the instance
        final_output_file = os.path.join(temp_output_dir, "output.jsonl")
        with open(final_output_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 1
        result = json.loads(lines[0])
        assert result["instance_id"] == "instance_1"

    def test_multiple_instances_with_mixed_results(self, temp_output_dir):
        """
        Test aggregation with multiple instances having different result patterns.

        Scenario:
        - instance_1: attempt 3 errors, attempt 2 non-error
        - instance_2: all attempts non-error, critic-fail
        - instance_3: attempt 2 critic-success, attempt 3 critic-fail

        Expected: All instances should appear in output.jsonl.
        """
        critic = FailCritic()

        # Attempt 1
        attempt_1_file = os.path.join(temp_output_dir, "output.critic_attempt_1.jsonl")
        with open(attempt_1_file, "w") as f:
            f.write(create_output("instance_1", error=None).model_dump_json() + "\n")
            f.write(create_output("instance_2", error=None).model_dump_json() + "\n")
            f.write(create_output("instance_3", error=None).model_dump_json() + "\n")

        # Attempt 2
        attempt_2_file = os.path.join(temp_output_dir, "output.critic_attempt_2.jsonl")
        with open(attempt_2_file, "w") as f:
            f.write(create_output("instance_1", error=None).model_dump_json() + "\n")
            f.write(create_output("instance_2", error=None).model_dump_json() + "\n")
            f.write(create_output("instance_3", error=None).model_dump_json() + "\n")

        # Attempt 3
        attempt_3_file = os.path.join(temp_output_dir, "output.critic_attempt_3.jsonl")
        with open(attempt_3_file, "w") as f:
            # instance_1 errors
            f.write(
                create_output("instance_1", error="Runtime error").model_dump_json()
                + "\n"
            )
            f.write(create_output("instance_2", error=None).model_dump_json() + "\n")
            f.write(create_output("instance_3", error=None).model_dump_json() + "\n")

        # Run aggregation
        aggregate_results(temp_output_dir, max_attempts=3, critic=critic)

        # Verify all instances appear in output.jsonl
        final_output_file = os.path.join(temp_output_dir, "output.jsonl")
        with open(final_output_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 3, "All 3 instances should appear in output"
        instance_ids = {json.loads(line)["instance_id"] for line in lines}
        assert instance_ids == {"instance_1", "instance_2", "instance_3"}

    def test_all_attempts_error_instance_dropped(self, temp_output_dir):
        """
        Test that instances where all attempts error are correctly dropped.

        If all attempts have errors, the instance should not appear in output.jsonl.
        """
        critic = PassCritic()

        # All attempts error
        for attempt in range(1, 4):
            attempt_file = os.path.join(
                temp_output_dir, f"output.critic_attempt_{attempt}.jsonl"
            )
            output = create_output("instance_1", error=f"Error in attempt {attempt}")
            with open(attempt_file, "w") as f:
                f.write(output.model_dump_json() + "\n")

        # Run aggregation
        aggregate_results(temp_output_dir, max_attempts=3, critic=critic)

        # Verify output.jsonl is empty (instance dropped because all attempts errored)
        final_output_file = os.path.join(temp_output_dir, "output.jsonl")
        with open(final_output_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 0, "Instance with all error attempts should be dropped"

    def test_empty_attempts(self, temp_output_dir):
        """Test aggregation when no attempt files exist."""
        critic = PassCritic()

        # Run aggregation with no attempt files
        aggregate_results(temp_output_dir, max_attempts=3, critic=critic)

        # Verify output.jsonl is created but empty
        final_output_file = os.path.join(temp_output_dir, "output.jsonl")
        assert os.path.exists(final_output_file)
        with open(final_output_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 0
