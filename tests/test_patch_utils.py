"""Tests for patch_utils module."""

from benchmarks.utils.patch_utils import remove_binary_diffs, remove_files_from_patch


class TestRemoveBinaryDiffs:
    """Tests for remove_binary_diffs function."""

    def test_no_binary_diffs(self):
        """Test that text diffs are preserved."""
        patch = """diff --git a/file.py b/file.py
index 1234567..89abcdef 100644
--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 line 1
+added line
 line 2
 line 3
"""
        result = remove_binary_diffs(patch)
        assert "added line" in result
        assert "Binary files" not in result

    def test_removes_binary_diff(self):
        """Test that binary file diffs are removed."""
        patch = """diff --git a/image.png b/image.png
index 1234567..89abcdef 100644
--- b/image.png
+++ b/image.png
Binary files differ
"""
        result = remove_binary_diffs(patch)
        assert result == ""

    def test_mixed_diffs_preserves_text(self):
        """Test that binary diffs are removed but text diffs are kept."""
        patch = """diff --git a/file.py b/file.py
index 1234567..89abcdef 100644
--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 line 1
+added line
 line 2

diff --git a/image.png b/image.png
index 1234567..89abcdef 100644
--- b/image.png
+++ b/image.png
Binary files differ
"""
        result = remove_binary_diffs(patch)
        assert "added line" in result
        assert "Binary files differ" not in result
        assert "image.png" not in result
        assert "file.py" in result

    def test_empty_patch(self):
        """Test that empty patches return empty string."""
        assert remove_binary_diffs("") == ""
        assert remove_binary_diffs(None) is None

    def test_binary_at_start(self):
        """Test patch with binary diff at start."""
        patch = """diff --git a/binary.bin b/binary.bin
index 1234567..89abcdef 100644
--- b/binary.bin
+++ b/binary.bin
Binary files differ

diff --git a/text.txt b/text.txt
index abcdefg..hijklmn 100644
--- a/text.txt
+++ b/text.txt
@@ -1 +1 @@
 hello
+world
"""
        result = remove_binary_diffs(patch)
        assert "text.txt" in result
        assert "world" in result
        assert "binary.bin" not in result

    def test_binary_at_end(self):
        """Test patch with binary diff at end."""
        patch = """diff --git a/text.txt b/text.txt
index abcdefg..hijklmn 100644
--- a/text.txt
+++ b/text.txt
@@ -1 +1 @@
 hello
+world

diff --git a/binary.bin b/binary.bin
index 1234567..89abcdef 100644
--- b/binary.bin
+++ b/binary.bin
Binary files differ
"""
        result = remove_binary_diffs(patch)
        assert "text.txt" in result
        assert "world" in result
        assert "binary.bin" not in result


class TestRemoveFilesFromPatch:
    """Tests for remove_files_from_patch function."""

    def test_removes_specified_file(self):
        """Test that specified files are removed from patch."""
        patch = """diff --git a/to_remove.py b/to_remove.py
index 1234567..89abcdef 100644
--- a/to_remove.py
+++ b/to_remove.py
@@ -1,3 +1,4 @@
 line 1
+removed content
 line 2
"""
        result = remove_files_from_patch(patch, ["to_remove.py"])
        assert result == ""

    def test_preserves_non_specified_files(self):
        """Test that non-specified files are preserved."""
        patch = """diff --git a/keep.py b/keep.py
index 1234567..89abcdef 100644
--- a/keep.py
+++ b/keep.py
@@ -1,3 +1,4 @@
 line 1
+keep this
 line 2
"""
        result = remove_files_from_patch(patch, ["other.py"])
        assert "keep this" in result

    def test_empty_patch(self):
        """Test that empty patches return empty string."""
        assert remove_files_from_patch("", ["file.py"]) == ""
        assert remove_files_from_patch(None, ["file.py"]) is None

    def test_multiple_files(self):
        """Test removing multiple files at once."""
        patch = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1 +1 @@
-old
+new

diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1 +1 @@
-old
+new
"""
        result = remove_files_from_patch(patch, ["file1.py", "file2.py"])
        assert result == ""
