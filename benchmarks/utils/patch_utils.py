"""
Utilities for handling patch generation in SWE-bench evaluation.
"""

from __future__ import annotations

import os
import re
from typing import Callable, List, Tuple


def remove_files_from_patch(git_patch, files):
    """
    Remove files modifications from a git patch string.
    Args:
        git_patch (str): The original git patch string
        files (List[str]): The files to remove form the patch
    Returns:
        str: The git patch with files modifications removed
    """
    if not git_patch:
        return git_patch

    # Split patch into individual file diffs
    # Look for diff --git patterns to identify file boundaries
    diff_pattern = r"diff --git [^\n]*\n"

    # Find all diff headers and their positions
    diff_matches = list(re.finditer(diff_pattern, git_patch))

    if not diff_matches:
        return git_patch

    # Extract individual file diffs
    file_diffs = []
    for i, match in enumerate(diff_matches):
        start = match.start()
        end = (
            diff_matches[i + 1].start() if i + 1 < len(diff_matches) else len(git_patch)
        )
        file_diff = git_patch[start:end]
        file_diffs.append(file_diff)

    # Filter out files in list
    filtered_diffs = []
    for diff in file_diffs:
        # Extract filenames from diff header to do exact matching
        should_skip = False
        if "diff --git" in diff:
            # Extract the diff header line
            first_line = diff.split("\n")[0]
            # Parse diff --git a/file b/file format
            match = re.match(r"diff --git a/(.+) b/(.+)", first_line)
            if match:
                file_a, file_b = match.groups()
                # Check if either filename (before or after) matches any file to remove
                if file_a in files or file_b in files:
                    should_skip = True

        if should_skip:
            # Skip this diff
            continue
        filtered_diffs.append(diff)

    # Rejoin the filtered diffs with proper newlines
    if not filtered_diffs:
        return ""

    # Join the diffs while preserving their original structure
    # Each diff already contains its proper ending from the original split
    result = "".join(filtered_diffs)

    return result


def remove_binary_diffs(patch_text):
    """
    Remove binary file diffs from a git patch.
    Args:
        patch_text (str): The git patch text
    Returns:
        str: The cleaned patch text with binary diffs removed
    """
    lines = patch_text.splitlines()
    cleaned_lines = []
    block = []
    is_binary_block = False

    for line in lines:
        if line.startswith("diff --git "):
            if block and not is_binary_block:
                cleaned_lines.extend(block)
            block = [line]
            is_binary_block = False
        elif "Binary files" in line:
            is_binary_block = True
            block.append(line)
        else:
            block.append(line)

    if block and not is_binary_block:
        cleaned_lines.extend(block)
    return "\n".join(cleaned_lines)


def remove_binary_files_from_git():
    """
    Generate a bash command to remove binary files from git staging.
    Returns:
        str: A bash command that removes binary files from git staging
    """
    return """
    for file in $(git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" | cut -c4-); do
        if [ -f "$file" ] && (file "$file" | grep -q "executable" || \\
            git check-attr binary "$file" | grep -q "binary: set"); then
            git rm -f "$file" 2>/dev/null || rm -f "$file"
            echo "Removed: $file"
        fi
    done
    """.strip()


def _iter_file_diffs(git_patch: str) -> List[Tuple[str, str, str]]:
    """
    Split a git patch into per-file chunks.

    Returns a list of tuples: (diff_text, file_a, file_b).
    """
    if not git_patch:
        return []

    diff_pattern = r"diff --git [^\n]*\n"
    diff_matches = list(re.finditer(diff_pattern, git_patch))
    if not diff_matches:
        return []

    file_diffs: List[Tuple[str, str, str]] = []
    for i, match in enumerate(diff_matches):
        start = match.start()
        end = (
            diff_matches[i + 1].start() if i + 1 < len(diff_matches) else len(git_patch)
        )
        diff = git_patch[start:end]
        header = diff.split("\n", 1)[0]
        file_a = file_b = ""
        m = re.match(r"diff --git a/(.+) b/(.+)", header)
        if m:
            file_a, file_b = m.groups()
        file_diffs.append((diff, file_a, file_b))
    return file_diffs


def filter_patch_by_predicate(
    git_patch: str, predicate: Callable[[str, str], bool]
) -> Tuple[str, int, int]:
    """
    Keep only file diffs that satisfy the predicate.

    Returns: (filtered_patch, kept_count, dropped_count).
    """
    file_diffs = _iter_file_diffs(git_patch)
    kept: List[str] = []
    dropped = 0
    for diff, file_a, file_b in file_diffs:
        if predicate(file_a, file_b):
            kept.append(diff)
        else:
            dropped += 1
    return ("".join(kept), len(kept), dropped)


def looks_like_test_file(path: str) -> bool:
    """
    Heuristic to decide whether a path represents a test file.
    """
    norm = path.lstrip("./")
    if not norm:
        return False

    blocked_prefixes = (
        "build/",
        "dist/",
        ".venv/",
        "venv/",
        "env/",
        ".tox/",
    )
    if norm.startswith(blocked_prefixes) or "/build/" in norm or "/dist/" in norm:
        return False

    name = os.path.basename(norm)
    if not name.endswith(".py"):
        return False

    if name in {"conftest.py", "tests.py"}:
        return True

    if name.startswith("test_") or name.endswith("_test.py"):
        return True

    if norm.startswith("tests/") or "/tests/" in norm:
        return True

    return False


def is_top_level(path: str) -> bool:
    """Return True if the path has no directory component (e.g., reproduction.py)."""
    norm = path.lstrip("./")
    return norm != "" and "/" not in norm


def remove_top_level_files(git_patch: str) -> Tuple[str, int, int]:
    """
    Drop diffs that touch top-level files.
    """
    return filter_patch_by_predicate(
        git_patch,
        lambda a, b: not (is_top_level(a) or is_top_level(b)),
    )
