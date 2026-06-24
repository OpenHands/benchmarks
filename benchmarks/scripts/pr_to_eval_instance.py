"""Emit a SWE-bench-style eval-instance row from a GitHub PR (ALL-2451).

Intended to run from the PR-merge capture job: clone/inspect the PR, then call
this to append an eval instance to a dataset. Reads a GitHub PR JSON payload
(as produced by ``gh pr view <n> --json number,title,body,baseRefOid,headRefOid,url``
or the REST API) from a file or stdin.

Examples:
    gh pr view 123 --repo owner/repo \\
        --json number,title,body,baseRefOid,headRefOid,url \\
        | python -m benchmarks.scripts.pr_to_eval_instance \\
            --repo owner/repo --base-commit <sha> --out instances.jsonl
"""

import argparse
import json
import sys

from benchmarks.utils.pr_snapshot import (
    build_swebench_instance,
    pr_snapshot_from_github_json,
    write_instances_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr-json",
        help="Path to a GitHub PR JSON payload (default: read stdin)",
    )
    parser.add_argument(
        "--repo", help="GitHub 'owner/repo' (overrides the payload if set)"
    )
    parser.add_argument(
        "--base-commit",
        help="Commit the agent started from; defaults to the PR base SHA",
    )
    parser.add_argument(
        "--patch-file", help="Path to the unified diff (base..head) for the PR"
    )
    parser.add_argument("--out", required=True, help="Destination .jsonl dataset file")
    parser.add_argument(
        "--append", action="store_true", help="Append to --out instead of overwriting"
    )
    args = parser.parse_args(argv)

    raw = (
        open(args.pr_json, encoding="utf-8").read()
        if args.pr_json
        else sys.stdin.read()
    )
    pr = json.loads(raw)
    if args.repo:
        pr["repo"] = args.repo

    patch = None
    if args.patch_file:
        patch = open(args.patch_file, encoding="utf-8").read()

    snapshot = pr_snapshot_from_github_json(
        pr, base_commit=args.base_commit, patch=patch
    )
    row = build_swebench_instance(snapshot)
    write_instances_jsonl([row], args.out, append=args.append)
    print(f"Wrote instance {row['instance_id']} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
