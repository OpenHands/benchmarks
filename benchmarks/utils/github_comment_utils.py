"""Utility functions for processing GitHub comments."""


def is_mention_in_unquoted_text(comment_body: str, mention: str) -> bool:
    """
    Check if a mention appears in unquoted (non-blockquote) text.

    GitHub's "Quote reply" feature creates markdown blockquotes with lines
    starting with '>'. This function checks if the mention appears outside
    of these quoted blocks.

    Args:
        comment_body: The full comment body text
        mention: The mention string to search for (e.g., '@openhands')

    Returns:
        True if the mention appears in unquoted text, False if it only
        appears in quoted blocks or doesn't appear at all.

    Examples:
        >>> body = "Hello @openhands please help"
        >>> is_mention_in_unquoted_text(body, "@openhands")
        True

        >>> body = "> Someone said @openhands\\nI agree"
        >>> is_mention_in_unquoted_text(body, "@openhands")
        False

        >>> body = "> Someone said @openhands\\n@openhands please help"
        >>> is_mention_in_unquoted_text(body, "@openhands")
        True
    """
    if not comment_body or not mention:
        return False

    # Split comment into lines
    lines = comment_body.split("\n")

    # Check each line
    for line in lines:
        stripped_line = line.lstrip()

        # Skip empty lines
        if not stripped_line:
            continue

        # Check if this is a quoted line (starts with >)
        is_quoted = stripped_line.startswith(">")

        # If the line is not quoted and contains the mention, return True
        if not is_quoted and mention in line:
            return True

    # If we get here, the mention either doesn't exist or only exists in quoted lines
    return False
