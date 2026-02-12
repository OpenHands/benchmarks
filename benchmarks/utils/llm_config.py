from __future__ import annotations

import json
import os
from pathlib import Path

from openhands.sdk import LLM


def load_llm_config(config_path: str | Path) -> LLM:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ValueError(f"LLM config file {config_path} does not exist")

    with config_path.open("r") as f:
        llm_config = json.load(f)

    # load api_key from env var if api_key_env is specified
    if "api_key_env" in llm_config:
        env_var = llm_config.pop("api_key_env")
        api_key = os.environ.get(env_var, "")
        if not api_key:
            raise ValueError(
                f"Environment variable {env_var} is not set or empty. "
                f"Please set it with your API key."
            )
        llm_config["api_key"] = api_key

    # strip /chat/completions from base_url for LiteLLM compatibility
    if "base_url" in llm_config:
        base_url = llm_config["base_url"]
        base_url = base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url.removesuffix("/chat/completions")
        llm_config["base_url"] = base_url

    return LLM.model_validate(llm_config)
