"""
Utilities for loading LLM configuration with environment variable overrides.
"""

import json
import os

from openhands.sdk import LLM


LLM_API_KEY_ENV_VAR = "LLM_API_KEY"


def load_llm_config(config_path: str) -> LLM:
    """Load LLM configuration from a JSON file with environment variable override.

    If the LLM_API_KEY environment variable is set, it will override the api_key
    value in the JSON configuration file. This allows cloud environments to inject
    the API key via secrets without modifying the config file.

    Args:
        config_path: Path to the JSON LLM configuration file.

    Returns:
        LLM instance with the loaded configuration.

    Raises:
        ValueError: If the config file does not exist.
    """
    if not os.path.isfile(config_path):
        raise ValueError(f"LLM config file {config_path} does not exist")

    with open(config_path, "r") as f:
        config_data = json.load(f)

    # Override api_key with environment variable if set
    env_api_key = os.getenv(LLM_API_KEY_ENV_VAR)
    if env_api_key:
        config_data["api_key"] = env_api_key

    return LLM.model_validate(config_data)
