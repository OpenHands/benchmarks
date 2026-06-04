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
        aggregate_results(temp_output_dir, n_critic_runs=3, critic=critic)

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
        aggregate_results(temp_output_dir, n_critic_runs=3, critic=critic)

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
        aggregate_results(temp_output_dir, n_critic_runs=3, critic=critic)

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
        aggregate_results(temp_output_dir, n_critic_runs=3, critic=critic)

        # Verify output.jsonl is empty (instance dropped because all attempts errored)
        final_output_file = os.path.join(temp_output_dir, "output.jsonl")
        with open(final_output_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 0, "Instance with all error attempts should be dropped"

    def test_empty_attempts(self, temp_output_dir):
        """Test aggregation when no attempt files exist."""
        critic = PassCritic()

        # Run aggregation with no attempt files
        aggregate_results(temp_output_dir, n_critic_runs=3, critic=critic)

        # Verify output.jsonl is created but empty
        final_output_file = os.path.join(temp_output_dir, "output.jsonl")
        assert os.path.exists(final_output_file)
        with open(final_output_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 0

    def test_preserves_entries_with_unknown_history_kinds(self, temp_output_dir):
        """
        Resumed runs may carry over critic-attempt entries whose ``history``
        references discriminated-union ``kind``s not registered in the current
        process (e.g. browser tools when the resume pod loads only the default
        toolset). Pydantic validation of those entries raises, and prior to
        this fix every such entry was silently dropped — emptying out
        ``output.jsonl`` and causing the eval phase to see zero instances.

        Verify that all non-error entries are preserved as-is, that error
        entries are still dropped, and that fully-parseable entries from a
        later attempt can still win the rank tie-break.
        """
        critic = PassCritic()
        attempt_1_file = os.path.join(temp_output_dir, "output.critic_attempt_1.jsonl")

        # Two carried-over rows whose history contains an unregistered action
        # kind, plus one carried-over row marked as an error.
        unparseable_ok_row = {
            "instance_id": "instance_ok",
            "instruction": "carried over",
            "instance": {"test": "data"},
            "test_result": {"git_patch": "carried patch"},
            "error": None,
            "history": [{"kind": "ThisActionKindIsNotRegisteredAnywhere"}],
        }
        unparseable_error_row = {
            "instance_id": "instance_err",
            "instruction": "carried over",
            "instance": {"test": "data"},
            "test_result": None,
            "error": "Timed out",
            "history": [{"kind": "ThisActionKindIsNotRegisteredAnywhere"}],
        }
        # And one cleanly parseable row from the resumed run's new attempt.
        clean_row = create_output("instance_new", error=None)

        with open(attempt_1_file, "w") as f:
            f.write(json.dumps(unparseable_ok_row) + "\n")
            f.write(json.dumps(unparseable_error_row) + "\n")
            f.write(clean_row.model_dump_json() + "\n")

        aggregate_results(temp_output_dir, n_critic_runs=3, critic=critic)

        final_output_file = os.path.join(temp_output_dir, "output.jsonl")
        with open(final_output_file, "r") as f:
            written = [json.loads(line) for line in f if line.strip()]

        written_ids = {row["instance_id"] for row in written}
        # The non-error unparseable carry-over MUST be preserved (regression
        # for the empty-output.jsonl bug observed on resumed gaia runs).
        assert "instance_ok" in written_ids
        # The cleanly parseable new attempt is included.
        assert "instance_new" in written_ids
        # The errored carry-over is still filtered out.
        assert "instance_err" not in written_ids
        assert len(written) == 2

        # Round-tripped carry-over must preserve the history (i.e. raw line
        # is written verbatim, not re-serialised through EvalOutput which
        # would silently drop the unknown kind).
        ok_row = next(row for row in written if row["instance_id"] == "instance_ok")
        assert ok_row["history"] == unparseable_ok_row["history"]

    def test_parseable_critic_success_beats_unparseable_carryover(
        self, temp_output_dir
    ):
        """
        A fully-parseable, critic-successful attempt for the same instance
        must out-rank the unparseable raw-line fallback so the freshest result
        wins, mirroring the pre-existing rank semantics for parseable rows.
        """
        critic = PassCritic()

        # Attempt 1: carried-over, unparseable, non-error → falls back to rank 1.
        attempt_1_file = os.path.join(temp_output_dir, "output.critic_attempt_1.jsonl")
        carryover = {
            "instance_id": "instance_1",
            "instruction": "carried",
            "instance": {"test": "data"},
            "test_result": {"git_patch": "old patch"},
            "error": None,
            "history": [{"kind": "ThisActionKindIsNotRegisteredAnywhere"}],
        }
        with open(attempt_1_file, "w") as f:
            f.write(json.dumps(carryover) + "\n")

        # Attempt 2: cleanly parseable, critic-successful (PassCritic) → rank 2.
        attempt_2_file = os.path.join(temp_output_dir, "output.critic_attempt_2.jsonl")
        fresh = create_output("instance_1", error=None)
        with open(attempt_2_file, "w") as f:
            f.write(fresh.model_dump_json() + "\n")

        aggregate_results(temp_output_dir, n_critic_runs=3, critic=critic)

        final_output_file = os.path.join(temp_output_dir, "output.jsonl")
        with open(final_output_file, "r") as f:
            written = [json.loads(line) for line in f if line.strip()]

        assert len(written) == 1
        # Fresh, critic-successful row wins → its test_result is kept.
        assert written[0]["test_result"] == {"git_patch": "mock patch"}
