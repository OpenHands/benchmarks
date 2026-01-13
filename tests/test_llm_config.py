"""Tests for LLM configuration loading with environment variable override."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from benchmarks.utils.llm_config import LLM_API_KEY_ENV_VAR, load_llm_config


def get_api_key_value(api_key: str | SecretStr | None) -> str | None:
    """Extract the actual value from api_key which can be str, SecretStr, or None."""
    if api_key is None:
        return None
    if isinstance(api_key, SecretStr):
        return api_key.get_secret_value()
    return api_key


@pytest.fixture
def sample_config():
    """Create a sample LLM config dict."""
    return {
        "model": "test-model",
        "base_url": "https://api.example.com",
        "api_key": "config-api-key",
    }


@pytest.fixture
def config_file(sample_config):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_config, f)
        f.flush()
        yield f.name
    os.unlink(f.name)


def test_load_llm_config_from_file(config_file, sample_config):
    """Test loading LLM config from file without env var override."""
    with patch.dict(os.environ, {}, clear=True):
        # Ensure LLM_API_KEY is not set
        os.environ.pop(LLM_API_KEY_ENV_VAR, None)

        llm = load_llm_config(config_file)

        assert llm.model == sample_config["model"]
        assert llm.base_url == sample_config["base_url"]
        assert get_api_key_value(llm.api_key) == sample_config["api_key"]


def test_load_llm_config_with_env_override(config_file, sample_config):
    """Test that LLM_API_KEY env var overrides the config file api_key."""
    env_api_key = "env-override-api-key"

    with patch.dict(os.environ, {LLM_API_KEY_ENV_VAR: env_api_key}):
        llm = load_llm_config(config_file)

        assert llm.model == sample_config["model"]
        assert llm.base_url == sample_config["base_url"]
        # api_key should be overridden by env var
        assert get_api_key_value(llm.api_key) == env_api_key


def test_load_llm_config_env_override_empty_string(config_file, sample_config):
    """Test that empty string env var does not override config."""
    with patch.dict(os.environ, {LLM_API_KEY_ENV_VAR: ""}):
        llm = load_llm_config(config_file)

        # Empty string is falsy, so config value should be used
        assert get_api_key_value(llm.api_key) == sample_config["api_key"]


def test_load_llm_config_file_not_found():
    """Test that ValueError is raised when config file doesn't exist."""
    with pytest.raises(ValueError, match="does not exist"):
        load_llm_config("/nonexistent/path/config.json")


def test_load_llm_config_without_api_key_in_file():
    """Test loading config without api_key in file, with env var set."""
    config_without_key = {
        "model": "test-model",
        "base_url": "https://api.example.com",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_without_key, f)
        f.flush()
        config_path = f.name

    try:
        env_api_key = "env-api-key"
        with patch.dict(os.environ, {LLM_API_KEY_ENV_VAR: env_api_key}):
            llm = load_llm_config(config_path)

            assert llm.model == config_without_key["model"]
            assert get_api_key_value(llm.api_key) == env_api_key
    finally:
        os.unlink(config_path)


def test_llm_api_key_env_var_constant():
    """Test that the env var constant is correctly defined."""
    assert LLM_API_KEY_ENV_VAR == "LLM_API_KEY"
