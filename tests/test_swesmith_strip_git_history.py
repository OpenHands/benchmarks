"""The SWE-smith sandbox must not expose the task's bug patch or clean code."""

import filecmp
import subprocess
from pathlib import Path

import pytest

from benchmarks.swesmith import build_images
from benchmarks.swesmith.run_infer import (
    replace_testbed_command,
    strip_git_history_command,
)
from benchmarks.utils.build_utils import BuildOutput


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

    # /testbed in the image: a full clone of the mirror, left on clean main.
    git(tmp_path, "clone", "-q", str(mirror), str(tmp_path / "testbed"))

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


def test_replace_testbed_leaves_only_the_stripped_task_tree(task_workspace):
    workspace, bug_commit = task_workspace
    testbed = workspace.parent / "testbed"
    assert "a + b" in (testbed / "lib.py").read_text()  # clean code before

    subprocess.run(
        ["bash", "-c", strip_git_history_command(str(workspace))], check=True
    )
    subprocess.run(
        ["bash", "-c", replace_testbed_command(str(workspace), str(testbed))],
        check=True,
    )

    assert "a - b" in (testbed / "lib.py").read_text()
    assert not (testbed / "test_lib.py").exists()
    assert git(testbed, "rev-list", "--all") == git(workspace, "rev-parse", "HEAD")
    assert git(testbed, "remote") == ""
    cmp = filecmp.dircmp(testbed, workspace, ignore=[])
    assert not (cmp.left_only or cmp.right_only or cmp.diff_files)


def test_built_images_get_the_testbed_access_layer(monkeypatch):
    calls = []

    def fake_layer(**kwargs):
        calls.append(kwargs)
        return BuildOutput(base_image="x", tags=kwargs["tags"])

    monkeypatch.setattr(build_images, "run_docker_build_layer", fake_layer)
    image = "ghcr.io/openhands/eval-agent-server:abc-swesmith.x86_64.r-source-minimal"
    built = BuildOutput(base_image="docker.io/swesmith.x86_64.r", tags=[image])

    assert build_images.wrap_testbed_access(built, push=True) is built
    assert calls[0]["dockerfile"] == build_images.TESTBED_ACCESS_DOCKERFILE
    assert calls[0]["tags"] == [image]
    assert calls[0]["build_args"] == {"SDK_IMAGE": image}
    assert calls[0]["push"] is True
    dockerfile = build_images.TESTBED_ACCESS_DOCKERFILE.read_text()
    assert "find /testbed -type d -exec chown openhands:openhands {} +" in dockerfile
    assert dockerfile.rstrip().endswith("USER openhands")
