"""Test for OpenAgentSafety 422 error fix."""

from pydantic import SecretStr

from benchmarks.openagentsafety.run_infer import create_server_compatible_llm
from openhands.sdk import LLM


def test_create_server_compatible_llm_removes_forbidden_fields():
    """Test that create_server_compatible_llm removes forbidden fields."""
    # Create an LLM with forbidden fields
    original_llm = LLM(
        model="test-model",
        api_key=SecretStr("test-key"),
        extra_headers={"X-Custom": "value"},
        reasoning_summary="detailed",
        litellm_extra_body={"custom": "data"},
        temperature=0.7,
    )

    # Create server-compatible version
    compatible_llm = create_server_compatible_llm(original_llm)

    # Verify forbidden fields are set to None/empty (effectively removed)
    compatible_data = compatible_llm.model_dump()
    assert compatible_data["extra_headers"] is None
    assert compatible_data["reasoning_summary"] is None
    assert compatible_data["litellm_extra_body"] == {}

    # Verify other fields are preserved
    assert compatible_data["model"] == "test-model"
    assert compatible_data["temperature"] == 0.7


def test_create_server_compatible_llm_handles_missing_fields():
    """Test that the function handles LLMs without forbidden fields gracefully."""
    # Create an LLM without forbidden fields
    original_llm = LLM(
        model="test-model",
        temperature=0.5,
    )

    # Create server-compatible version
    compatible_llm = create_server_compatible_llm(original_llm)

    # Verify it works without errors
    compatible_data = compatible_llm.model_dump()
    assert compatible_data["model"] == "test-model"
    assert compatible_data["temperature"] == 0.5


def test_create_server_compatible_llm_preserves_secrets():
    """Test that secrets are properly handled during the conversion."""
    # Create an LLM with secrets
    original_llm = LLM(
        model="test-model",
        api_key=SecretStr("secret-key"),
        aws_access_key_id=SecretStr("aws-key"),
        extra_headers={"X-Custom": "value"},  # This should be removed
    )

    # Create server-compatible version
    compatible_llm = create_server_compatible_llm(original_llm)

    # Verify secrets are preserved
    assert compatible_llm.api_key is not None
    assert compatible_llm.aws_access_key_id is not None

    # Verify forbidden field is set to None (effectively removed)
    compatible_data = compatible_llm.model_dump()
    assert compatible_data["extra_headers"] is None
