"""LiteLLM proxy virtual key management for per-instance cost tracking.

Uses LiteLLM virtual keys to get exact per-instance costs from the proxy
instead of relying on token-count-based estimation. Each eval instance gets
its own virtual key so the proxy tracks spend independently.

Requires:
    - LLM_BASE_URL: The LiteLLM proxy URL (existing env var)
    - LLM_API_MASTER_KEY: Admin key for virtual key management (SOPS secret)

When either is unset, all functions are no-ops.

Thread-safety:
    The virtual key for the current instance is stored in a ``threading.local``
    so that concurrent worker threads (asyncio.to_thread) each track their own
    key without global state mutation. ``build_acp_agent`` in ``acp.py`` reads
    this thread-local to inject the key via ``acp_env``.
"""

import os
import threading

import httpx
from pydantic import SecretStr

from openhands.sdk import LLM, get_logger


logger = get_logger(__name__)

_TIMEOUT = 30.0

# Thread-local storage for the current instance's virtual key.
# Each worker thread sets this before evaluate_instance() and clears it after.
_thread_local = threading.local()


def _get_config() -> tuple[str, str] | None:
    """Return (base_url, master_key) or None if not configured."""
    base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
    master_key = os.getenv("LLM_API_MASTER_KEY", "")
    if not base_url or not master_key:
        return None
    return base_url, master_key


def create_virtual_key(
    instance_id: str,
    run_id: str | None = None,
    max_budget: float = 50.0,
) -> str | None:
    """Create a LiteLLM virtual key for tracking per-instance spend.

    Args:
        instance_id: Evaluation instance identifier (stored in key metadata).
        run_id: Optional evaluation run identifier.
        max_budget: Safety budget cap in USD per instance.

    Returns:
        The virtual key string, or None if proxy is not configured.

    Raises:
        RuntimeError: If proxy is configured but key creation fails.
    """
    config = _get_config()
    if config is None:
        return None
    base_url, api_key = config

    metadata: dict[str, str] = {"instance_id": instance_id}
    if run_id:
        metadata["run_id"] = run_id

    try:
        resp = httpx.post(
            f"{base_url}/key/generate",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"metadata": metadata, "max_budget": max_budget},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        key = resp.json()["key"]
        logger.info("[litellm-proxy] Created virtual key for instance %s", instance_id)
        return key
    except Exception as e:
        raise RuntimeError(
            f"LiteLLM proxy is configured but virtual key creation failed "
            f"for {instance_id}: {e}"
        ) from e


def get_key_spend(key: str) -> float | None:
    """Query actual USD spend tracked by the proxy for a virtual key.

    Args:
        key: The virtual key string.

    Returns:
        Spend in USD, or None on failure.
    """
    config = _get_config()
    if config is None:
        return None
    base_url, api_key = config

    try:
        resp = httpx.get(
            f"{base_url}/key/info",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"key": key},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        spend = resp.json()["info"]["spend"]
        logger.info("[litellm-proxy] Key spend: $%.6f", spend)
        return float(spend)
    except Exception as e:
        logger.warning("[litellm-proxy] Failed to query key spend: %s", e)
        return None


def delete_key(key: str) -> None:
    """Delete a virtual key from the proxy.

    Args:
        key: The virtual key string to delete.
    """
    config = _get_config()
    if config is None:
        return
    base_url, api_key = config

    try:
        resp = httpx.post(
            f"{base_url}/key/delete",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"keys": [key]},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        logger.debug("[litellm-proxy] Deleted virtual key")
    except Exception as e:
        logger.warning("[litellm-proxy] Failed to delete virtual key: %s", e)


def set_current_virtual_key(key: str | None) -> None:
    """Store a virtual key in thread-local storage for the current worker.

    Called by the evaluation orchestrator before ``evaluate_instance()``.
    ``build_acp_agent()`` in ``acp.py`` reads this to inject the key into
    the ACP subprocess environment via ``acp_env``.
    """
    _thread_local.virtual_key = key


def get_current_virtual_key() -> str | None:
    """Return the virtual key for the current worker thread, or None."""
    return getattr(_thread_local, "virtual_key", None)


def build_eval_llm(llm: LLM, *, usage_id: str | None = None) -> LLM:
    """Return an LLM configured for the current evaluation instance.

    The default (non-ACP) OpenHands agent talks to LiteLLM in-process, so it
    must use the thread-local per-instance virtual key to record exact proxy
    spend for that instance.

    We also force ``extended_thinking_budget=0`` here: the SDK LLM class
    defaults this to 200_000, which for Anthropic models triggers the
    ``interleaved-thinking-2025-05-14`` beta header (see
    ``openhands/sdk/llm/options/chat_options.py``). Interleaved thinking is
    mutually exclusive with parallel tool calls on the Anthropic API — the
    model can never emit more than one tool_use block per assistant message
    while that beta is active. Disabling it here is required for
    ``tool_concurrency_limit>1`` on the agent to have any observable effect.
    """
    updates: dict[str, object] = {"extended_thinking_budget": 0}
    if usage_id is not None:
        updates["usage_id"] = usage_id

    virtual_key = get_current_virtual_key()
    if virtual_key is not None:
        updates["api_key"] = SecretStr(virtual_key)

    return llm.model_copy(deep=True, update=updates)
