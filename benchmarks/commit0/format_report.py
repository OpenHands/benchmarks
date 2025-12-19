#!/usr/bin/env python3
"""
Commit0 Report Formatter

Generates a markdown notification message from Commit0 evaluation results.
The message is used for Slack notifications and PR comments.

Usage:
    python format_report.py <output.jsonl> <report.json> [--env-file <env_file>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_env_file(env_file: str) -> None:
    """Populate environment variables from a simple KEY=VALUE file."""
    env_path = Path(env_file)
    if not env_path.exists():
        return
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key] = value.strip().strip('"').strip("'")


def format_commit0_report(
    report: dict[str, Any],
    eval_name: str,
    model_name: str,
    dataset: str,
    dataset_split: str,
    repo_split: str,
    commit: str,
    timestamp: str,
    trigger_reason: str | None,
    tar_url: str | None,
) -> str:
    """Format Commit0 evaluation results as markdown."""
    total_instances = report.get("total_instances", 0)
    submitted_instances = report.get("submitted_instances", 0)
    resolved_instances = report.get("resolved_instances", 0)
    unresolved_instances = report.get("unresolved_instances", 0)
    empty_patch_instances = report.get("empty_patch_instances", 0)
    error_instances = report.get("error_instances", 0)
    total_tests = report.get("total_tests", 0)
    total_passed_tests = report.get("total_passed_tests", 0)

    success_rate = "N/A"
    if submitted_instances > 0:
        pct = (resolved_instances / submitted_instances) * 100
        success_rate = f"{resolved_instances}/{submitted_instances} ({pct:.1f}%)"

    lines = [
        "## 🎉 Commit0 Evaluation Complete",
        "",
        f"**Evaluation:** `{eval_name}`",
        f"**Model:** `{model_name}`",
        f"**Dataset:** `{dataset}` (`{dataset_split}`, repo split `{repo_split}`)",
        f"**Commit:** `{commit}`",
        f"**Timestamp:** {timestamp}",
    ]

    if trigger_reason:
        lines.append(f"**Reason:** {trigger_reason}")

    lines.extend(
        [
            "",
            "### 📊 Results",
            f"- **Total instances:** {total_instances}",
            f"- **Submitted instances:** {submitted_instances}",
            f"- **Resolved instances:** {resolved_instances}",
            f"- **Unresolved instances:** {unresolved_instances}",
            f"- **Empty patch instances:** {empty_patch_instances}",
            f"- **Error instances:** {error_instances}",
            f"- **Tests passed:** {total_passed_tests}/{total_tests}",
            f"- **Success rate:** {success_rate}",
        ]
    )

    if tar_url:
        lines.extend(
            [
                "",
                "### 🔗 Links",
                f"[Full Archive]({tar_url})",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format Commit0 evaluation results for notifications"
    )
    parser.add_argument("output_jsonl", help="Path to output.jsonl from evaluation")
    parser.add_argument(
        "report_json", help="Path to report.json with aggregated metrics"
    )
    parser.add_argument(
        "--env-file",
        help="Optional env file containing metadata (UNIQUE_EVAL_NAME, MODEL_NAME, etc.)",
    )
    args = parser.parse_args()

    if args.env_file:
        load_env_file(args.env_file)

    try:
        report = load_json(args.report_json)
    except Exception as exc:
        print(f"Error loading report.json: {exc}", file=sys.stderr)
        sys.exit(1)

    eval_name = os.environ.get("UNIQUE_EVAL_NAME", "unknown")
    model_name = os.environ.get("MODEL_NAME", "unknown")
    dataset = os.environ.get("DATASET", "commit0")
    dataset_split = os.environ.get("DATASET_SPLIT", "test")
    repo_split = os.environ.get("REPO_SPLIT", "lite")
    commit = os.environ.get("COMMIT", "unknown")
    timestamp = os.environ.get("TIMESTAMP", "unknown")
    trigger_reason = os.environ.get("TRIGGER_REASON")
    tar_url = os.environ.get("TAR_URL")

    message = format_commit0_report(
        report=report,
        eval_name=eval_name,
        model_name=model_name,
        dataset=dataset,
        dataset_split=dataset_split,
        repo_split=repo_split,
        commit=commit,
        timestamp=timestamp,
        trigger_reason=trigger_reason,
        tar_url=tar_url,
    )

    print(message)


if __name__ == "__main__":
    main()
