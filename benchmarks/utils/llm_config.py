from __future__ import annotations

from pathlib import Path

from openhands.sdk import LLM


def load_llm_config(config_path: str | Path) -> LLM:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ValueError(f"LLM config file {config_path} does not exist")

    with config_path.open("r") as f:
        llm_config = f.read()

    return LLM.model_validate_json(llm_config)
