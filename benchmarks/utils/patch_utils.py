"""
Utilities for handling patch generation in SWE-bench evaluation.
"""

import re


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


def _is_test_path(path: str) -> bool:
    """Heuristic: does ``path`` look like a project test file?

    A path counts as a test if it lives under a ``tests``/``test``/``testing``
    directory anywhere in its tree, or if its basename matches the standard
    pytest/unittest discovery patterns (``test_*.py``, ``*_test.py``,
    ``conftest.py``).

    Files at the repository root are *not* considered tests even when they
    match the naming pattern (e.g. ``test_repro.py``), because in SWT-bench
    those are agent-authored scratch files that the harness ignores.
    """
    if "/" not in path:
        return False
    parts = path.split("/")
    if any(seg in ("tests", "test", "testing") for seg in parts[:-1]):
        return True
    base = parts[-1]
    return (
        (base.startswith("test_") and base.endswith(".py")) or base.endswith("_test.py") or base == "conftest.py"
    )


def keep_only_test_files(git_patch: str) -> str:
    """Return ``git_patch`` with every non-test file diff removed.

    Useful for benchmarks that only score test-file changes (e.g. SWT-bench).
    A file is kept iff :func:`_is_test_path` returns ``True`` for either the
    pre- or post-image path in its ``diff --git`` header.

    The implementation mirrors :func:`remove_files_from_patch` so the two
    helpers behave consistently.
    """
    if not git_patch:
        return git_patch

    diff_pattern = r"diff --git [^\n]*\n"
    diff_matches = list(re.finditer(diff_pattern, git_patch))
    if not diff_matches:
        return git_patch

    kept = []
    for i, match in enumerate(diff_matches):
        start = match.start()
        end = (
            diff_matches[i + 1].start() if i + 1 < len(diff_matches) else len(git_patch)
        )
        diff = git_patch[start:end]
        header = diff.split("\n", 1)[0]
        m = re.match(r"diff --git a/(.+) b/(.+)", header)
        if m and (_is_test_path(m.group(1)) or _is_test_path(m.group(2))):
            kept.append(diff)

    return "".join(kept)


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
