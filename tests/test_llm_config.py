"""Tests for load_llm_config utility.

This function is used by all 7 benchmarks, so comprehensive tests
are critical to prevent regressions.
"""

import json
import os
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from benchmarks.utils.llm_config import load_llm_config
from openhands.sdk import LLM


class TestLoadLLMConfigValidConfigs:
    """Test that valid JSON config files load correctly."""

    def test_minimal_valid_config(self, tmp_path: Path) -> None:
        """Minimal config with only required 'model' field loads correctly."""
        config = {"model": "gpt-4o"}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        llm = load_llm_config(config_path)

        assert isinstance(llm, LLM)
        assert llm.model == "gpt-4o"

    def test_full_valid_config(self, tmp_path: Path) -> None:
        """Config with all common fields loads correctly."""
        config = {
            "model": "litellm_proxy/anthropic/claude-sonnet-4-20250514",
            "base_url": "https://llm-proxy.eval.all-hands.dev",
            "api_key": "test-api-key",
            "temperature": 0.7,
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        llm = load_llm_config(config_path)

        assert llm.model == "litellm_proxy/anthropic/claude-sonnet-4-20250514"
        assert llm.base_url == "https://llm-proxy.eval.all-hands.dev"
        # api_key is a SecretStr, need to get the actual value
        assert llm.api_key is not None
        assert isinstance(llm.api_key, SecretStr)
        assert llm.api_key.get_secret_value() == "test-api-key"
        assert llm.temperature == 0.7

    def test_config_with_string_path(self, tmp_path: Path) -> None:
        """Config path can be passed as string."""
        config = {"model": "gpt-3.5-turbo"}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        llm = load_llm_config(str(config_path))

        assert llm.model == "gpt-3.5-turbo"

    def test_config_with_path_object(self, tmp_path: Path) -> None:
        """Config path can be passed as Path object."""
        config = {"model": "gpt-4o"}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        llm = load_llm_config(config_path)

        assert llm.model == "gpt-4o"


class TestLoadLLMConfigMissingFile:
    """Test that missing files raise ValueError with appropriate message."""

    def test_missing_file_raises_value_error(self) -> None:
        """Non-existent file raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            load_llm_config("/nonexistent/path/config.json")

        assert "does not exist" in str(exc_info.value)

    def test_missing_file_error_includes_path(self, tmp_path: Path) -> None:
        """Error message includes the missing file path."""
        missing_path = tmp_path / "missing_config.json"

        with pytest.raises(ValueError) as exc_info:
            load_llm_config(missing_path)

        assert str(missing_path) in str(exc_info.value)

    def test_directory_instead_of_file_raises_value_error(self, tmp_path: Path) -> None:
        """Directory path instead of file raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            load_llm_config(tmp_path)

        assert "does not exist" in str(exc_info.value)


class TestLoadLLMConfigMalformedJSON:
    """Test that malformed JSON raises clear validation errors."""

    def test_invalid_json_syntax(self, tmp_path: Path) -> None:
        """Invalid JSON syntax raises ValidationError (via pydantic)."""
        config_path = tmp_path / "config.json"
        config_path.write_text("{invalid json}")

        # Pydantic's model_validate_json raises ValidationError for invalid JSON
        with pytest.raises(ValidationError) as exc_info:
            load_llm_config(config_path)
        assert "json" in str(exc_info.value).lower()

    def test_json_with_trailing_comma(self, tmp_path: Path) -> None:
        """JSON with trailing comma raises ValidationError."""
        config_path = tmp_path / "config.json"
        config_path.write_text('{"model": "gpt-4",}')

        with pytest.raises(ValidationError) as exc_info:
            load_llm_config(config_path)
        assert "json" in str(exc_info.value).lower()

    def test_json_with_unquoted_key(self, tmp_path: Path) -> None:
        """JSON with unquoted key raises ValidationError."""
        config_path = tmp_path / "config.json"
        config_path.write_text('{model: "gpt-4"}')

        with pytest.raises(ValidationError) as exc_info:
            load_llm_config(config_path)
        assert "json" in str(exc_info.value).lower()

    def test_missing_required_model_field(self, tmp_path: Path) -> None:
        """JSON without required 'model' field raises ValidationError."""
        config = {"temperature": 0.7}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        with pytest.raises(ValidationError) as exc_info:
            load_llm_config(config_path)

        # Check error mentions the missing field
        error_str = str(exc_info.value)
        assert "model" in error_str.lower()

    def test_invalid_field_type(self, tmp_path: Path) -> None:
        """JSON with wrong field type raises ValidationError."""
        config = {"model": "gpt-4", "temperature": "not-a-number"}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        with pytest.raises(ValidationError) as exc_info:
            load_llm_config(config_path)

        assert "temperature" in str(exc_info.value).lower()


class TestLoadLLMConfigEdgeCases:
    """Test edge cases like empty files and permissions issues."""

    def test_empty_file_raises_error(self, tmp_path: Path) -> None:
        """Empty file raises appropriate error."""
        config_path = tmp_path / "config.json"
        config_path.write_text("")

        # Pydantic's model_validate_json raises ValidationError for empty input
        with pytest.raises(ValidationError) as exc_info:
            load_llm_config(config_path)
        assert "json" in str(exc_info.value).lower()

    def test_whitespace_only_file_raises_error(self, tmp_path: Path) -> None:
        """File with only whitespace raises appropriate error."""
        config_path = tmp_path / "config.json"
        config_path.write_text("   \n\t   ")

        with pytest.raises(ValidationError) as exc_info:
            load_llm_config(config_path)
        assert "json" in str(exc_info.value).lower()

    def test_valid_json_but_not_object(self, tmp_path: Path) -> None:
        """Valid JSON that's not an object raises ValidationError."""
        config_path = tmp_path / "config.json"
        config_path.write_text('["model", "gpt-4"]')

        with pytest.raises(ValidationError):
            load_llm_config(config_path)

    def test_json_null_raises_error(self, tmp_path: Path) -> None:
        """JSON null value raises ValidationError."""
        config_path = tmp_path / "config.json"
        config_path.write_text("null")

        with pytest.raises(ValidationError):
            load_llm_config(config_path)

    @pytest.mark.skipif(
        os.name == "nt", reason="File permissions behave differently on Windows"
    )
    def test_unreadable_file_raises_permission_error(self, tmp_path: Path) -> None:
        """File without read permissions raises PermissionError."""
        config_path = tmp_path / "config.json"
        config_path.write_text('{"model": "gpt-4"}')
        config_path.chmod(0o000)

        try:
            with pytest.raises(PermissionError):
                load_llm_config(config_path)
        finally:
            # Restore permissions for cleanup
            config_path.chmod(0o644)

    def test_config_with_extra_fields_loads(self, tmp_path: Path) -> None:
        """Config with unknown extra fields should still load (pydantic default)."""
        config = {
            "model": "gpt-4o",
            "unknown_field": "value",
            "another_unknown": 123,
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        # Should not raise - pydantic by default ignores extra fields
        llm = load_llm_config(config_path)
        assert llm.model == "gpt-4o"

    def test_unicode_in_config(self, tmp_path: Path) -> None:
        """Config with unicode characters loads correctly."""
        config = {"model": "gpt-4o", "api_key": "key-with-émojis-🔑"}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False))

        llm = load_llm_config(config_path)
        assert llm.api_key is not None
        assert isinstance(llm.api_key, SecretStr)
        assert llm.api_key.get_secret_value() == "key-with-émojis-🔑"
