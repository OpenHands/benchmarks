from __future__ import annotations

import json
from pathlib import Path

from benchmarks.utils.intelligent_routing import (
    RouterSpec,
    is_router_config_payload,
    maybe_load_router_spec,
)
from openhands.sdk import LLM


def load_llm_config(config_path: str | Path) -> LLM:
    """Load an SDK :class:`LLM` from a JSON config file.

    For backwards compatibility, this function also accepts an intelligent
    router config (``kind: intelligent-router-v0``); in that case it returns
    the classifier LLM, which downstream code uses as the "primary" LLM (e.g.
    for ACP agents, condensers, or as the fallback when routing is bypassed).
    Use :func:`maybe_load_router_spec` to additionally retrieve the routing
    configuration.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ValueError(f"LLM config file {config_path} does not exist")

    text = config_path.read_text(encoding="utf-8")

    # Fast path: plain LLM config (the overwhelmingly common case). Avoid the
    # double-parse for non-router configs by sniffing only when the JSON
    # parses to an object carrying our discriminator.
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Let pydantic produce its existing ValidationError on malformed JSON.
        return LLM.model_validate_json(text)

    if is_router_config_payload(payload):
        spec: RouterSpec = maybe_load_router_spec(config_path)  # type: ignore[assignment]
        # Surface the classifier LLM as the "primary" LLM so callers that
        # only need an LLM (e.g. ACP wiring) continue to work.
        return spec.classifier_llm

    return LLM.model_validate(payload)


__all__ = ["load_llm_config", "maybe_load_router_spec"]
