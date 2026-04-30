from __future__ import annotations

from pathlib import Path

from openhands.sdk import LLM
from openhands.sdk.llm.utils.model_features import model_matches


# Models where LiteLLM handles reasoning_effort incorrectly.
# LiteLLM maps reasoning_effort="high" to type="adaptive" for 4.6 but to
# type="enabled" with fixed budget_tokens=4096 for 4.7, causing issues.
OPUS_4_7_MODELS = [
    "claude-opus-4-7",
]


def load_llm_config(config_path: str | Path) -> LLM:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ValueError(f"LLM config file {config_path} does not exist")

    with config_path.open("r", encoding="utf-8") as f:
        llm_config = f.read()

    llm = LLM.model_validate_json(llm_config)

    # FIX: LiteLLM handles reasoning_effort differently for Opus 4.6 vs 4.7.
    # For 4.6, reasoning_effort="high" maps to type="adaptive" (model decides).
    # For 4.7, it maps to type="enabled" with fixed budget_tokens=4096.
    # This causes unexpected behavior (excessive thinking, token limit issues).
    # The fix: disable reasoning_effort for Opus 4.7 models to use default behavior.
    if model_matches(llm.model, OPUS_4_7_MODELS) and llm.reasoning_effort is not None:
        llm = LLM(
            **{
                **llm.model_dump(),
                "reasoning_effort": None,
            }
        )

    return llm
