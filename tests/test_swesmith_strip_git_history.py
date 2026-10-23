"""The SWE-smith workspace must not expose the task's bug patch through git."""

import subprocess
from pathlib import Path

import pytest

from benchmarks.swesmith.run_infer import strip_git_history_command


TASK = "owner__repo.abc123.combine_file__task1"
OTHER_TASK = "owner__repo.abc123.combine_file__task2"


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def task_workspace(tmp_path: Path) -> tuple[Path, str]:
    """A workspace laid out like run_infer leaves it before the agent starts.

    The mirror has a clean main and two task branches, each with a
    "Bug Patch" and a "Remove F2P Tests" commit. The workspace is a full
    clone (so every task branch is a remote ref, as in the SWE-smith image)
    with the task branch fetched and checked out detached.
    """
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    git(mirror, "init", "-q", "-b", "main")
    (mirror / "lib.py").write_text("def add(a, b):\n    return a + b\n")
    (mirror / "test_lib.py").write_text("assert add(1, 2) == 3\n")
    git(mirror, "add", "-A")
    git(mirror, "commit", "-q", "-m", "Initial commit")
    for branch in (TASK, OTHER_TASK):
        git(mirror, "checkout", "-q", "-b", branch, "main")
        (mirror / "lib.py").write_text("def add(a, b):\n    return a - b\n")
        git(mirror, "commit", "-q", "-am", "Bug Patch")
        git(mirror, "rm", "-q", "test_lib.py")
        git(mirror, "commit", "-q", "-m", "Remove F2P Tests")
    git(mirror, "checkout", "-q", "main")

    workspace = tmp_path / "workspace"
    git(tmp_path, "clone", "-q", str(mirror), str(workspace))
    git(workspace, "fetch", "-q", str(mirror), TASK)
    git(workspace, "checkout", "-q", "FETCH_HEAD")
    bug_commit = git(workspace, "rev-parse", "HEAD~1")
    return workspace, bug_commit


def test_strip_git_history_keeps_only_the_task_tree(task_workspace):
    workspace, bug_commit = task_workspace
    tree_before = git(workspace, "rev-parse", "HEAD^{tree}")

    subprocess.run(
        ["bash", "-c", strip_git_history_command(str(workspace))], check=True
    )

    assert git(workspace, "rev-list", "--all").splitlines() == [
        git(workspace, "rev-parse", "HEAD")
    ]
    assert git(workspace, "rev-parse", "HEAD^{tree}") == tree_before
    assert git(workspace, "status", "--porcelain") == ""
    assert git(workspace, "remote") == ""
    assert git(workspace, "reflog") == ""
    missing = subprocess.run(
        ["git", "cat-file", "-e", bug_commit], cwd=workspace, capture_output=True
    )
    assert missing.returncode != 0, "bug patch commit is still in the object store"


def test_strip_git_history_patch_against_new_base_is_unchanged(task_workspace):
    """eval_infer applies `git diff <base_commit> HEAD`; the agent's edit must
    produce the same patch as before the history was stripped."""
    workspace, _ = task_workspace
    subprocess.run(
        ["bash", "-c", strip_git_history_command(str(workspace))], check=True
    )
    base = git(workspace, "rev-parse", "HEAD")

    (workspace / "lib.py").write_text("def add(a, b):\n    return a + b\n")
    git(workspace, "add", "-A")
    git(workspace, "commit", "-q", "-m", "patch")
    patch = git(workspace, "diff", base, "HEAD")

    assert patch.startswith("diff --git a/lib.py b/lib.py")
    assert "-    return a - b\n+    return a + b" in patch
