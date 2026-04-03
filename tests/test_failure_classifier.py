"""Tests for benchmarks.utils.failure_classifier."""

import pytest

from benchmarks.utils.failure_classifier import FailureCategory, classify_failure


class TestClassifyFailure:
    """classify_failure should return the correct category for known patterns."""

    # -- Non-resource failures -------------------------------------------------

    @pytest.mark.parametrize(
        "message",
        [
            "ACPPromptError: terminated",
            "ACPPromptError: Internal Server Error",
            "ACPPromptError: Model stream ended with empty response text.",
            "ACPPromptError: Model stream ended with malformed function call",
            "ACP prompt timed out after 3600.1s",
            "ACP error: terminated",
            "ACP error: Internal Server Error",
            "HTTP request failed (503 Service Unavailable)",
            "Server disconnected without sending a response",
            "Remote conversation ended with error",
            "Conversation run failed for id=abc: Remote conversation ended with error",
            "malformed function call in response",
            "temp and top_p cannot both be specified",
            "does not support parameters: ['reasoning_effort']",
        ],
    )
    def test_non_resource_failures(self, message: str) -> None:
        assert classify_failure(Exception(message)) == FailureCategory.NON_RESOURCE

    # -- Resource failures -----------------------------------------------------

    @pytest.mark.parametrize(
        "message",
        [
            "Agent server image ghcr.io/foo/bar:latest does not exist in container registry",
            "ImagePullBackOff for container agent-server",
            "ErrImagePull: rpc error",
            "Runtime not yet ready after 300s",
            "OOMKilled",
            "OutOfMemory",
            "Pod cannot be scheduled",
            "Insufficient cpu",
            "Insufficient memory",
        ],
    )
    def test_resource_failures(self, message: str) -> None:
        assert classify_failure(Exception(message)) == FailureCategory.RESOURCE

    # -- Unknown failures default to RESOURCE ----------------------------------

    def test_unknown_failure_defaults_to_resource(self) -> None:
        assert classify_failure(Exception("some novel error")) == FailureCategory.RESOURCE

    # -- Chained exceptions ----------------------------------------------------

    def test_chained_exception_inner_match(self) -> None:
        """Non-resource pattern in __cause__ should be detected."""
        outer = RuntimeError("Conversation run failed")
        inner = Exception("ACPPromptError: terminated")
        outer.__cause__ = inner
        assert classify_failure(outer) == FailureCategory.NON_RESOURCE

    def test_chained_exception_outer_match(self) -> None:
        """Pattern in the outer exception should also match."""
        outer = RuntimeError("ACP prompt timed out after 3600s")
        inner = Exception("some detail")
        outer.__cause__ = inner
        assert classify_failure(outer) == FailureCategory.NON_RESOURCE

    # -- Wrapped "Remote conversation ended with error" ------------------------

    def test_wrapped_remote_conversation_error(self) -> None:
        """Common wrapping pattern from eval_infer."""
        err = RuntimeError(
            "Conversation run failed for id=abc-123: "
            "Remote conversation ended with error"
        )
        assert classify_failure(err) == FailureCategory.NON_RESOURCE
