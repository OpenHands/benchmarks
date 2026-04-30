"""Tests for SWE-Bench eval_infer functionality."""

import json
import tempfile

from benchmarks.swebench.eval_infer import convert_to_swebench_format
from benchmarks.utils.constants import MODEL_NAME_OR_PATH


class TestConvertToSwebenchFormat:
    """Tests for convert_to_swebench_format function."""

    def test_empty_input_file_does_not_raise(self):
        """Test that an empty input file does not raise an exception.

        When no entries are converted, the script should continue normally
        rather than raising an exception. The harness is responsible for
        handling empty results appropriately.
        """
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
        convert_to_swebench_format(input_path, output_path)

        # Verify output file is empty
        with open(output_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 0

    def test_model_name_or_path_uses_constant(self):
        """Test that model_name_or_path uses the MODEL_NAME_OR_PATH constant."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as infile:
            # Write a valid entry
            entry = {
                "instance_id": "django__django-12345",
                "test_result": {"git_patch": "diff --git a/test.py b/test.py"},
            }
            infile.write(json.dumps(entry) + "\n")
            input_path = infile.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".swebench.jsonl", delete=False
        ) as outfile:
            output_path = outfile.name

        convert_to_swebench_format(input_path, output_path)

        with open(output_path, "r") as f:
            result = json.loads(f.readline())

        assert result["model_name_or_path"] == MODEL_NAME_OR_PATH

    def test_binary_diffs_are_removed(self):
        """Test that binary diffs are removed from the patch."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as infile:
            # Patch with a binary diff that should be removed
            patch_with_binary = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1 +1 @@
-old
+new
diff --git a/binary.png b/binary.png
Binary files differ
diff --git a/another.py b/another.py
--- a/another.py
+++ a/another.py
@@ -1 +1 @@
-foo
+bar"""
            entry = {
                "instance_id": "django__django-12345",
                "test_result": {"git_patch": patch_with_binary},
            }
            infile.write(json.dumps(entry) + "\n")
            input_path = infile.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".swebench.jsonl", delete=False
        ) as outfile:
            output_path = outfile.name

        convert_to_swebench_format(input_path, output_path)

        with open(output_path, "r") as f:
            result = json.loads(f.readline())

        # Verify binary diff was removed
        assert "Binary files differ" not in result["model_patch"]
        assert "diff --git a/binary.png" not in result["model_patch"]
        # Verify regular diffs are preserved
        assert "diff --git a/test.py" in result["model_patch"]
        assert "diff --git a/another.py" in result["model_patch"]
