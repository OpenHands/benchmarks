"""Utilities for formatting model names in benchmark reports."""

from benchmarks.utils.version import SDK_SHORT_SHA


def format_model_name_or_path(model_name: str | None) -> str:
    """Format model_name_or_path for benchmark reports.

    Args:
        model_name: Optional model identifier from LLM config (e.g.,
            "litellm_proxy/claude-sonnet-4-5-20250929").

    Returns:
        Formatted string:
        - "OpenHands-{SDK_SHORT_SHA}/{extracted_model_name}" if model_name is provided
        - "OpenHands-{SDK_SHORT_SHA}" if model_name is None or empty

    Examples:
        >>> format_model_name_or_path("litellm_proxy/claude-sonnet-4-5-20250929")
        'OpenHands-5967796/claude-sonnet-4-5-20250929'
        >>> format_model_name_or_path("claude-sonnet-4-5-20250929")
        'OpenHands-5967796/claude-sonnet-4-5-20250929'
        >>> format_model_name_or_path(None)
        'OpenHands-5967796'
        >>> format_model_name_or_path("")
        'OpenHands-5967796'
    """
    base = f"OpenHands-{SDK_SHORT_SHA}"
    if model_name:
        extracted_model_name = model_name.rsplit("/", 1)[-1]
        return f"{base}/{extracted_model_name}"
    return base
