#!/usr/bin/env python3
"""
Script to check if a GitHub comment contains an unquoted mention.

This script is designed to be used in GitHub Actions workflows to prevent
triggering the bot when mentions only appear in quoted reply blocks.

Usage:
    python check_mention.py <comment_body> <mention>

Exit codes:
    0: Mention found in unquoted text (should trigger bot)
    1: Mention not found or only in quoted text (should not trigger bot)
"""

import sys

from benchmarks.utils.github_comment_utils import is_mention_in_unquoted_text


def main():
    """Check if mention is in unquoted text and return appropriate exit code."""
    if len(sys.argv) != 3:
        print("Usage: check_mention.py <comment_body> <mention>", file=sys.stderr)
        sys.exit(2)

    comment_body = sys.argv[1]
    mention = sys.argv[2]

    # Check if mention is in unquoted text
    if is_mention_in_unquoted_text(comment_body, mention):
        print(f"Mention '{mention}' found in unquoted text - should trigger bot")
        sys.exit(0)
    else:
        print(
            f"Mention '{mention}' not found or only in quoted text - should not trigger bot"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
