"""Utilities for formatting model names in benchmark reports."""

from benchmarks.utils.version import SDK_SHORT_SHA


def format_model_name_or_path(model_name: str) -> str:
    """Format model_name_or_path for benchmark reports.

    Args:
        model_name: Model identifier from LLM config (e.g.,
            "litellm_proxy/claude-sonnet-4-5-20250929").

    Returns:
        Formatted string: "OpenHands-{SDK_SHORT_SHA}/{extracted_model_name}"

    Raises:
        ValueError: If model_name is empty or None.

    Examples:
        >>> format_model_name_or_path("litellm_proxy/claude-sonnet-4-5-20250929")
        'OpenHands-5967796/claude-sonnet-4-5-20250929'
        >>> format_model_name_or_path("claude-sonnet-4-5-20250929")
        'OpenHands-5967796/claude-sonnet-4-5-20250929'
    """
    if not model_name:
        raise ValueError(
            "model_name is required. Provide the LLM model identifier "
            "(e.g., 'litellm_proxy/claude-sonnet-4-5-20250929')."
        )
    extracted_model_name = model_name.rsplit("/", 1)[-1]
    return f"OpenHands-{SDK_SHORT_SHA}/{extracted_model_name}"
