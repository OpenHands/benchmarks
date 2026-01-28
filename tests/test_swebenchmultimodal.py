"""Tests for SWE-Bench Multimodal eval_infer functionality."""

import tempfile

from benchmarks.swebenchmultimodal.eval_infer import convert_to_swebench_format


class TestConvertToSwebenchFormat:
    """Tests for convert_to_swebench_format function."""

    def test_empty_input_file_does_not_raise(self):
        """Test that an empty input file does not raise an exception."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as infile:
            infile.write("")  # Empty file
            input_path = infile.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".swebench.jsonl", delete=False
        ) as outfile:
            output_path = outfile.name

        # Should not raise - let the harness handle empty results
        convert_to_swebench_format(
            input_path, output_path, "litellm_proxy/claude-sonnet-4-5-20250929"
        )

        # Verify output file is empty
        with open(output_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 0

    def test_raises_when_model_name_missing(self):
        """Ensure a missing model identifier is rejected."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as infile:
            infile.write("")
            input_path = infile.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".swebench.jsonl", delete=False
        ) as outfile:
            output_path = outfile.name

        try:
            convert_to_swebench_format(input_path, output_path, None)  # type: ignore[arg-type]
        except ValueError:
            return

        assert False, "Expected ValueError when model_name_or_path is None"
