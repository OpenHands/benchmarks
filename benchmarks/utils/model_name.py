"""Utilities for formatting model names in benchmark reports."""


def format_model_name_or_path(model_name: str | None) -> str:
    """Format model_name_or_path for benchmark reports.

    Args:
        model_name: Optional model identifier from LLM config (e.g.,
            "litellm_proxy/claude-sonnet-4-5-20250929").

    Returns:
        Formatted string:
        - "OpenHands/{extracted_model_name}" if model_name is provided
        - "OpenHands" if model_name is None or empty

    Examples:
        >>> format_model_name_or_path("litellm_proxy/claude-sonnet-4-5-20250929")
        'OpenHands/claude-sonnet-4-5-20250929'
        >>> format_model_name_or_path("claude-sonnet-4-5-20250929")
        'OpenHands/claude-sonnet-4-5-20250929'
        >>> format_model_name_or_path(None)
        'OpenHands'
        >>> format_model_name_or_path("")
        'OpenHands'
    """
    if model_name:
        extracted_model_name = model_name.rsplit("/", 1)[-1]
        return f"OpenHands/{extracted_model_name}"
    return "OpenHands"
