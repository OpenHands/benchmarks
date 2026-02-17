"""Tests for the validate_git_reset script."""

import tempfile
from pathlib import Path

from benchmarks.scripts.validate_git_reset import (
    find_git_operations,
    has_git_reset_nearby,
    validate_file,
)


class TestFindGitOperations:
    def test_finds_git_clone(self):
        content = """
some code
git clone https://example.com/repo.git
more code
"""
        operations = find_git_operations(content)
        assert len(operations) == 1
        assert operations[0][0] == 3  # line number
        assert "git clone" in operations[0][1]  # line content
        assert operations[0][2] == "git clone"  # operation type

    def test_finds_git_checkout(self):
        content = """
some code
git checkout main
more code
"""
        operations = find_git_operations(content)
        assert len(operations) == 1
        assert operations[0][0] == 3
        assert "git checkout" in operations[0][1]
        assert operations[0][2] == "git checkout"

    def test_ignores_git_checkout_b(self):
        """git checkout -b creates a new branch, so doesn't need reset."""
        content = """
some code
git checkout -b new-branch
more code
"""
        operations = find_git_operations(content)
        assert len(operations) == 0

    def test_finds_multiple_operations(self):
        content = """
git clone https://repo1.git
some code
git clone https://repo2.git
git checkout feature
"""
        operations = find_git_operations(content)
        assert len(operations) == 3

    def test_finds_operations_in_strings(self):
        content = """
action = CmdRunAction(command=f'git clone {repo_url}')
"""
        operations = find_git_operations(content)
        assert len(operations) == 1


class TestHasGitResetNearby:
    def test_finds_reset_after_operation(self):
        content = """git clone https://example.com/repo.git
git reset --hard HEAD
"""
        assert has_git_reset_nearby(content, 1) is True

    def test_finds_reset_in_comment_after(self):
        content = """git clone https://example.com/repo.git
# git reset is not needed here
"""
        assert has_git_reset_nearby(content, 1) is True

    def test_finds_reset_in_comment_before(self):
        content = """# git reset is not needed here because this is a utility
git checkout main
"""
        assert has_git_reset_nearby(content, 2) is True

    def test_no_reset_nearby(self):
        content = """git clone https://example.com/repo.git
some other code
more code
"""
        assert has_git_reset_nearby(content, 1, context_lines_after=2) is False

    def test_reset_beyond_context_window(self):
        content = """git clone https://example.com/repo.git
line 1
line 2
line 3
line 4
line 5
git reset --hard HEAD
"""
        # Default context is 20 lines, so this should be found
        assert has_git_reset_nearby(content, 1) is True
        # With small context window, should not be found
        assert has_git_reset_nearby(content, 1, context_lines_after=3) is False


class TestValidateFile:
    def test_valid_file_with_reset(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
clone_cmd = 'git clone https://example.com/repo.git'
os.system(clone_cmd)
os.system('git reset --hard HEAD')
""")
            f.flush()
            violations = validate_file(Path(f.name))
            assert len(violations) == 0

    def test_valid_file_with_comment(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
# git reset is not needed here - this is a dev environment utility
cmd = 'git checkout main'
os.system(cmd)
""")
            f.flush()
            violations = validate_file(Path(f.name))
            assert len(violations) == 0

    def test_invalid_file_missing_reset(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
clone_cmd = 'git clone https://example.com/repo.git'
os.system(clone_cmd)
# no reset here
""")
            f.flush()
            violations = validate_file(Path(f.name))
            assert len(violations) == 1
            assert violations[0][2] == "git clone"

    def test_shell_script_with_reset(self):
        with tempfile.NamedTemporaryFile(suffix=".sh", mode="w", delete=False) as f:
            f.write("""#!/bin/bash
git clone https://example.com/repo.git /workspace/repo
cd /workspace/repo && git reset --hard HEAD
""")
            f.flush()
            violations = validate_file(Path(f.name))
            assert len(violations) == 0
