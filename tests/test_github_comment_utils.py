"""Tests for GitHub comment utility functions."""

from benchmarks.utils.github_comment_utils import is_mention_in_unquoted_text


class TestIsMentionInUnquotedText:
    """Tests for is_mention_in_unquoted_text function."""

    def test_direct_mention_only(self):
        """Test a direct mention without any quotes."""
        body = "Hello @openhands please help with this issue"
        assert is_mention_in_unquoted_text(body, "@openhands") is True

    def test_quoted_mention_only(self):
        """Test a mention that only appears in a quoted block."""
        body = "> Someone said @openhands earlier\n> Can you help?"
        assert is_mention_in_unquoted_text(body, "@openhands") is False

    def test_mixed_mention_quoted_and_direct(self):
        """Test a mention in both quoted and unquoted text."""
        body = "> Someone said @openhands earlier\n\nYes, @openhands please help!"
        assert is_mention_in_unquoted_text(body, "@openhands") is True

    def test_multiple_quote_levels(self):
        """Test nested quotes (>> for double-quoted)."""
        body = ">> Original: @openhands\n> Reply: @openhands\nMy response"
        assert is_mention_in_unquoted_text(body, "@openhands") is False

    def test_mention_after_quote_on_same_line(self):
        """Test mention on a quoted line (should not trigger)."""
        body = "> Please @openhands help with this"
        assert is_mention_in_unquoted_text(body, "@openhands") is False

    def test_no_mention(self):
        """Test when mention doesn't appear at all."""
        body = "This is a regular comment without any mentions"
        assert is_mention_in_unquoted_text(body, "@openhands") is False

    def test_empty_body(self):
        """Test with empty comment body."""
        assert is_mention_in_unquoted_text("", "@openhands") is False

    def test_empty_mention(self):
        """Test with empty mention string."""
        body = "Hello world"
        assert is_mention_in_unquoted_text(body, "") is False

    def test_quote_with_spaces(self):
        """Test quoted lines with leading spaces before >."""
        body = "  > Quoted text with @openhands\nNormal text"
        assert is_mention_in_unquoted_text(body, "@openhands") is False

    def test_multiline_with_direct_mention_first(self):
        """Test direct mention followed by quoted mention."""
        body = "@openhands please help\n> Someone else said @openhands too"
        assert is_mention_in_unquoted_text(body, "@openhands") is True

    def test_github_quote_reply_example(self):
        """Test realistic GitHub quote reply scenario."""
        # Simulates what happens when someone clicks "Quote reply"
        original_comment = "@openhands fix this bug"
        quoted_reply = f"> {original_comment}\n\nI second this request"

        # Should not trigger because mention is only in the quote
        assert is_mention_in_unquoted_text(quoted_reply, "@openhands") is False

    def test_github_quote_reply_with_new_mention(self):
        """Test quote reply where user also adds a direct mention."""
        quoted_reply = "> @openhands fix this bug\n\n@openhands please also check this"

        # Should trigger because there's a direct mention
        assert is_mention_in_unquoted_text(quoted_reply, "@openhands") is True

    def test_different_mention_string(self):
        """Test with a different mention string."""
        body = "@openhands-agent please help"
        assert is_mention_in_unquoted_text(body, "@openhands-agent") is True

        body = "> @openhands-agent was mentioned"
        assert is_mention_in_unquoted_text(body, "@openhands-agent") is False

    def test_mention_in_code_block_not_quoted(self):
        """Test mention in code block (should still trigger as it's not a quote)."""
        body = "```python\n# Call @openhands\n```"
        # This should trigger because it's not a quoted line (>)
        assert is_mention_in_unquoted_text(body, "@openhands") is True

    def test_blank_lines_between_quotes(self):
        """Test handling of blank lines between quoted sections."""
        body = "> First quote @openhands\n\n> Second quote"
        assert is_mention_in_unquoted_text(body, "@openhands") is False

    def test_mention_multiple_times_in_unquoted(self):
        """Test multiple mentions in unquoted text."""
        body = "@openhands please help. CC @openhands-team"
        assert is_mention_in_unquoted_text(body, "@openhands") is True
