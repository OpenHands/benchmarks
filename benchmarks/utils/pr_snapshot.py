"""Build SWE-bench-style eval instances from PR-linked task captures (ALL-2451).

When an OpenHands Cloud conversation produces a pull request, we want to turn it
into an evaluation instance: the initial repo state the agent started from
(``repo`` + ``base_commit``), the task (``problem_statement``), and the change
that resulted (``patch``). That makes the conversation replayable as an eval.

This module is the deterministic data-shaping kernel: given the PR metadata and
the captured base commit, it produces a row in the same shape the SWE-bench
harness already consumes (see ``benchmarks/swebench/run_infer.py``, which reads
``instance.data['repo']`` / ``['base_commit']`` / ``['problem_statement']`` and
loads datasets from JSONL via ``get_dataset``).

The PR-merge trigger and the clone/delta-storage job (a Kubernetes job started
when a PR moves to MERGED/CLOSED) are deployment concerns tracked separately;
this module is what that job calls to emit each instance.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class PRSnapshot(BaseModel):
    """A PR-linked task capture, sufficient to recreate the initial eval state."""

    repo: str = Field(..., description="GitHub 'owner/repo'")
    pr_number: int = Field(..., description="Pull request number")
    base_commit: str = Field(
        ..., description="Commit the agent started from (initial repo state)"
    )
    title: str = Field(default="", description="PR title")
    body: str = Field(default="", description="PR description / task text")
    head_commit: str | None = Field(
        default=None, description="Tip commit of the PR branch (final state)"
    )
    patch: str | None = Field(
        default=None,
        description="Unified diff base_commit..head (the resulting change)",
    )
    html_url: str | None = Field(default=None, description="PR URL, for provenance")


def _instance_id(repo: str, pr_number: int) -> str:
    """SWE-bench-style id, e.g. 'django__django-11333'."""
    return f"{repo.replace('/', '__')}-{pr_number}"


def build_swebench_instance(snapshot: PRSnapshot) -> dict[str, Any]:
    """Return a SWE-bench-style dataset row for ``snapshot``.

    The row carries the fields the SWE-bench harness reads (``instance_id``,
    ``repo``, ``base_commit``, ``problem_statement``) plus the resulting
    ``patch`` and provenance metadata. It is JSON-serialisable and can be
    written to a ``.jsonl`` dataset consumable by ``get_dataset``.
    """
    problem_statement = snapshot.title
    if snapshot.body:
        problem_statement = (
            f"{snapshot.title}\n\n{snapshot.body}" if snapshot.title else snapshot.body
        )
    return {
        "instance_id": _instance_id(snapshot.repo, snapshot.pr_number),
        "repo": snapshot.repo,
        "base_commit": snapshot.base_commit,
        "problem_statement": problem_statement,
        "patch": snapshot.patch or "",
        "hints_text": "",
        "pr_url": snapshot.html_url,
        "head_commit": snapshot.head_commit,
        "created_from": "openhands-cloud-pr",
    }


def pr_snapshot_from_github_json(
    pr: dict[str, Any], base_commit: str | None = None, patch: str | None = None
) -> PRSnapshot:
    """Build a ``PRSnapshot`` from a GitHub PR payload (``gh pr view --json ...``).

    ``base_commit`` defaults to the PR's base SHA, but callers that captured the
    exact commit the agent started from should pass it explicitly — a PR's base
    ref can advance after the conversation started.
    """
    base = pr.get("baseRefOid") or {}
    base_sha = base_commit or (base if isinstance(base, str) else "")
    if not base_sha:
        # REST shape: {"base": {"sha": ...}, "head": {"sha": ...}}
        base_sha = (pr.get("base") or {}).get("sha", "")
    head = pr.get("headRefOid")
    if not head:
        head = (pr.get("head") or {}).get("sha")

    repo = pr.get("repo") or _repo_from_url(pr.get("html_url") or pr.get("url") or "")
    number = pr.get("number")
    if number is None:
        raise ValueError("PR payload missing 'number'")

    return PRSnapshot(
        repo=repo,
        pr_number=int(number),
        base_commit=base_sha,
        title=pr.get("title", "") or "",
        body=pr.get("body", "") or "",
        head_commit=head,
        patch=patch,
        html_url=pr.get("html_url") or pr.get("url"),
    )


def _repo_from_url(url: str) -> str:
    """Extract 'owner/repo' from a GitHub PR URL."""
    parts = [p for p in url.split("/") if p]
    # .../github.com/<owner>/<repo>/pull/<n>
    if "pull" in parts:
        i = parts.index("pull")
        if i >= 2:
            return f"{parts[i - 2]}/{parts[i - 1]}"
    return ""


def write_instances_jsonl(
    rows: list[dict[str, Any]], path: str | Path, append: bool = False
) -> None:
    """Write SWE-bench-style rows to a ``.jsonl`` dataset file."""
    mode = "a" if append else "w"
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
