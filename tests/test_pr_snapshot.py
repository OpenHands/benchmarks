"""Tests for benchmarks.utils.pr_snapshot (ALL-2451)."""

import json

from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.pr_snapshot import (
    PRSnapshot,
    build_swebench_instance,
    pr_snapshot_from_github_json,
    write_instances_jsonl,
)


def test_build_instance_has_swebench_fields():
    snap = PRSnapshot(
        repo="OpenHands/OpenHands",
        pr_number=123,
        base_commit="abc123",
        title="Fix the bug",
        body="Steps to reproduce ...",
        head_commit="def456",
        patch="diff --git a/x b/x\n",
        html_url="https://github.com/OpenHands/OpenHands/pull/123",
    )

    row = build_swebench_instance(snap)

    assert row["instance_id"] == "OpenHands__OpenHands-123"
    assert row["repo"] == "OpenHands/OpenHands"
    assert row["base_commit"] == "abc123"
    assert row["problem_statement"] == "Fix the bug\n\nSteps to reproduce ..."
    assert row["patch"] == "diff --git a/x b/x\n"
    assert row["pr_url"].endswith("/pull/123")
    assert row["created_from"] == "openhands-cloud-pr"


def test_build_instance_without_body_uses_title_only():
    snap = PRSnapshot(repo="o/r", pr_number=7, base_commit="c0", title="Title only")
    row = build_swebench_instance(snap)
    assert row["problem_statement"] == "Title only"
    assert row["patch"] == ""


def test_from_github_graphql_json():
    pr = {
        "number": 42,
        "title": "Add feature",
        "body": "Do the thing",
        "baseRefOid": "base-sha",
        "headRefOid": "head-sha",
        "url": "https://github.com/o/r/pull/42",
        "repo": "o/r",
    }

    snap = pr_snapshot_from_github_json(pr)

    assert snap.repo == "o/r"
    assert snap.pr_number == 42
    assert snap.base_commit == "base-sha"
    assert snap.head_commit == "head-sha"


def test_from_github_rest_json_and_repo_from_url():
    pr = {
        "number": 9,
        "title": "t",
        "body": "",
        "base": {"sha": "rest-base"},
        "head": {"sha": "rest-head"},
        "html_url": "https://github.com/acme/widgets/pull/9",
    }

    snap = pr_snapshot_from_github_json(pr)

    assert snap.repo == "acme/widgets"  # derived from the URL
    assert snap.base_commit == "rest-base"
    assert snap.head_commit == "rest-head"


def test_explicit_base_commit_overrides_payload():
    pr = {"number": 1, "title": "t", "baseRefOid": "stale", "repo": "o/r"}
    snap = pr_snapshot_from_github_json(pr, base_commit="captured-sha")
    assert snap.base_commit == "captured-sha"


def test_write_jsonl_roundtrips_through_get_dataset(tmp_path):
    rows = [
        build_swebench_instance(
            PRSnapshot(repo="o/r", pr_number=1, base_commit="c1", title="one")
        ),
        build_swebench_instance(
            PRSnapshot(repo="o/r", pr_number=2, base_commit="c2", title="two")
        ),
    ]
    out = tmp_path / "instances.jsonl"
    write_instances_jsonl(rows, out)

    # Plain JSONL is readable line-by-line ...
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["instance_id"] == "o__r-1"

    # ... and loads through the harness's dataset loader.
    df = get_dataset(str(out), split="train")
    assert set(df["instance_id"]) == {"o__r-1", "o__r-2"}


def test_write_jsonl_append(tmp_path):
    out = tmp_path / "instances.jsonl"
    write_instances_jsonl(
        [build_swebench_instance(PRSnapshot(repo="o/r", pr_number=1, base_commit="c"))],
        out,
    )
    write_instances_jsonl(
        [build_swebench_instance(PRSnapshot(repo="o/r", pr_number=2, base_commit="c"))],
        out,
        append=True,
    )
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 2
