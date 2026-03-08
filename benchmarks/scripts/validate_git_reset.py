"""Validation script to ensure git reset follows git clone/checkout operations.

During benchmark evaluation, we want to test a git repository at a specific commit.
To prevent the agent from looking at commits that are not part of the benchmark,
this script validates that every `git clone` or `git checkout` is followed by a
`git reset` command (or has a comment indicating git reset is not needed).

Usage:
    validate-git-reset [path...]

If no paths are provided, defaults to scanning the repository root.
"""

import argparse
import re
import sys
from pathlib import Path


# Patterns for finding git clone/checkout commands
GIT_CLONE_PATTERN = re.compile(r"git\s+clone\b")
GIT_CHECKOUT_PATTERN = re.compile(r"git\s+checkout\b")

# Pattern for git reset (can be in code or in a comment)
GIT_RESET_PATTERN = re.compile(r"git\s+reset\b")

# File extensions to check
CHECK_EXTENSIONS = {".py", ".sh"}

# Files to skip (relative to repository root)
SKIP_PATTERNS = [
    "validate_git_reset.py",  # This script itself
    "test_validate_git_reset.py",  # Test file for this script
]


def should_skip_file(file_path: Path) -> bool:
    """Check if a file should be skipped from validation."""
    for pattern in SKIP_PATTERNS:
        if file_path.name == pattern or pattern in str(file_path):
            return True
    return False


def find_git_operations(
    content: str,
) -> list[tuple[int, str, str]]:
    """Find all git clone and git checkout operations in the content.

    Returns a list of tuples: (line_number, line_content, operation_type)
    """
    operations = []
    lines = content.split("\n")

    for i, line in enumerate(lines):
        line_num = i + 1  # 1-indexed line numbers
        if GIT_CLONE_PATTERN.search(line):
            operations.append((line_num, line, "git clone"))
        # Only flag git checkout if it's not "git checkout -b" (creating a branch)
        # which doesn't need a reset since it's creating a new branch
        elif GIT_CHECKOUT_PATTERN.search(line):
            # Skip "git checkout -b" (create branch) as it doesn't need reset
            if not re.search(r"git\s+checkout\s+-b\b", line):
                operations.append((line_num, line, "git checkout"))

    return operations


def has_git_reset_nearby(
    content: str,
    operation_line: int,
    context_lines_after: int = 20,
    context_lines_before: int = 5,
) -> bool:
    """Check if there's a git reset within context lines around the operation.

    Also accepts git reset in comments as valid (to allow explicit documentation
    that reset is intentionally skipped).
    """
    lines = content.split("\n")
    start_line = max(0, operation_line - 1 - context_lines_before)
    end_line = min(operation_line + context_lines_after, len(lines))

    # Check lines before and after the operation
    for i in range(start_line, end_line):
        if GIT_RESET_PATTERN.search(lines[i]):
            return True

    return False


def validate_file(file_path: Path) -> list[tuple[int, str, str]]:
    """Validate a single file for git reset after clone/checkout.

    Returns a list of violations: (line_number, line_content, operation_type)
    """
    violations = []

    try:
        content = file_path.read_text()
    except (OSError, UnicodeDecodeError):
        return violations

    operations = find_git_operations(content)

    for line_num, line_content, op_type in operations:
        if not has_git_reset_nearby(content, line_num):
            violations.append((line_num, line_content.strip(), op_type))

    return violations


def find_files_to_check(paths: list[Path]) -> list[Path]:
    """Find all files that should be checked for git operations."""
    files = []

    for path in paths:
        if path.is_file():
            if path.suffix in CHECK_EXTENSIONS and not should_skip_file(path):
                files.append(path)
        elif path.is_dir():
            for ext in CHECK_EXTENSIONS:
                for file in path.rglob(f"*{ext}"):
                    if not should_skip_file(file):
                        files.append(file)

    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate that git reset follows git clone/checkout operations"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path(".")],
        help="Paths to check (files or directories). Defaults to current directory.",
    )
    args = parser.parse_args()

    files = find_files_to_check(args.paths)

    all_violations: list[tuple[Path, int, str, str]] = []

    for file in files:
        violations = validate_file(file)
        for line_num, line_content, op_type in violations:
            all_violations.append((file, line_num, line_content, op_type))

    if all_violations:
        print("ERROR: Found git clone/checkout without git reset:", file=sys.stderr)
        print(file=sys.stderr)
        for file, line_num, line_content, op_type in all_violations:
            print(f"  {file}:{line_num}: {op_type}", file=sys.stderr)
            print(f"    {line_content}", file=sys.stderr)
            print(file=sys.stderr)
        print(
            "To fix: Add 'git reset --hard <commit>' after the git operation,",
            file=sys.stderr,
        )
        print(
            "or add a comment containing 'git reset' to indicate it's intentional.",
            file=sys.stderr,
        )
        print(
            "Example: # git reset is not needed here because...",
            file=sys.stderr,
        )
        return 1

    print(f"OK: Checked {len(files)} files, no violations found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
