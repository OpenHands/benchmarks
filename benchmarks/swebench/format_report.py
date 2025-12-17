#!/usr/bin/env python3
"""
SWE-bench Report Formatter

Generates a unified markdown notification message from SWE-bench evaluation results.
This message is used for both Slack notifications and GitHub PR comments.

Usage:
    python format_report.py <output.jsonl> <report.json> [--env-file <env_file>]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load JSONL file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def format_swe_bench_report(
    report: dict[str, Any],
    eval_name: str,
    model_name: str,
    dataset: str,
    dataset_split: str,
    commit: str,
    timestamp: str,
    trigger_reason: str | None = None,
    tar_url: str | None = None,
) -> str:
    """
    Format SWE-bench evaluation results as a markdown notification.

    Args:
        report: SWE-bench report dictionary with metrics
        eval_name: Unique evaluation name
        model_name: Model name used
        dataset: Dataset name
        dataset_split: Dataset split used
        commit: Commit SHA
        timestamp: Evaluation timestamp
        trigger_reason: Optional reason for triggering the evaluation
        tar_url: URL to full results archive

    Returns:
        Markdown formatted notification message
    """
    # Extract SWE-bench metrics
    total_instances = report.get("total_instances", 0)
    submitted_instances = report.get("submitted_instances", 0)
    resolved_instances = report.get("resolved_instances", 0)
    unresolved_instances = report.get("unresolved_instances", 0)
    empty_patch_instances = report.get("empty_patch_instances", 0)
    error_instances = report.get("error_instances", 0)

    # Format success rate
    success_rate = "N/A"
    if submitted_instances > 0:
        success_rate = (
            f"{resolved_instances}/{submitted_instances} "
            f"({(resolved_instances / submitted_instances) * 100:.1f}%)"
        )

    # Build markdown message
    lines = [
        "## 🎉 SWE-bench Evaluation Complete",
        "",
        f"**Evaluation:** `{eval_name}`",
        f"**Model:** `{model_name}`",
        f"**Dataset:** `{dataset}` (`{dataset_split}`)",
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
            f"- **Success rate:** {success_rate}",
        ]
    )

    # Add link to full archive if available
    if tar_url:
        lines.extend(
            [
                "",
                "### 🔗 Links",
                f"[Full Archive]({tar_url})",
            ]
        )

    return "\n".join(lines)


def format_swe_bench_failure(
    eval_name: str,
    model_name: str,
    dataset: str,
    dataset_split: str,
    commit: str,
    timestamp: str,
    error_message: str,
    trigger_reason: str | None = None,
) -> str:
    """
    Format SWE-bench evaluation failure notification.

    Args:
        eval_name: Unique evaluation name
        model_name: Model name used
        dataset: Dataset name
        dataset_split: Dataset split used
        commit: Commit SHA
        timestamp: Evaluation timestamp
        error_message: Error details
        trigger_reason: Optional reason for triggering the evaluation

    Returns:
        Markdown formatted failure notification
    """
    lines = [
        "## ❌ SWE-bench Evaluation Failed",
        "",
        f"**Evaluation:** `{eval_name}`",
        f"**Model:** `{model_name}`",
        f"**Dataset:** `{dataset}` (`{dataset_split}`)",
        f"**Commit:** `{commit}`",
        f"**Timestamp:** {timestamp}",
    ]

    if trigger_reason:
        lines.append(f"**Reason:** {trigger_reason}")

    lines.extend(
        [
            "",
            "### ⚠️ Error Details",
            "```",
            error_message or "See logs for details",
            "```",
        ]
    )

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Format SWE-bench evaluation results for notifications"
    )
    parser.add_argument(
        "output_jsonl",
        help="Path to output.jsonl from evaluation",
    )
    parser.add_argument(
        "report_json",
        help="Path to report.json with aggregated metrics",
    )
    parser.add_argument(
        "--env-file",
        help="Optional environment file with evaluation metadata",
    )
    parser.add_argument(
        "--output",
        help="Output file for formatted message (default: stdout)",
    )

    args = parser.parse_args()

    # Load report.json for aggregated metrics
    try:
        report = load_json(args.report_json)
    except Exception as e:
        print(f"Error loading report.json: {e}", file=sys.stderr)
        sys.exit(1)

    # Load environment variables (from file or environment)
    if args.env_file and Path(args.env_file).exists():
        # Load from file if provided
        with open(args.env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key] = value.strip('"').strip("'")

    # Get required environment variables
    eval_name = os.environ.get("UNIQUE_EVAL_NAME", "unknown")
    model_name = os.environ.get("MODEL_NAME", "unknown")
    dataset = os.environ.get("DATASET", "swe-bench")
    dataset_split = os.environ.get("DATASET_SPLIT", "test")
    commit = os.environ.get("COMMIT", "unknown")
    timestamp = os.environ.get("TIMESTAMP", "unknown")

    # Optional variables
    trigger_reason = os.environ.get("TRIGGER_REASON")
    tar_url = os.environ.get("TAR_URL")

    # Format the message
    message = format_swe_bench_report(
        report=report,
        eval_name=eval_name,
        model_name=model_name,
        dataset=dataset,
        dataset_split=dataset_split,
        commit=commit,
        timestamp=timestamp,
        trigger_reason=trigger_reason,
        tar_url=tar_url,
    )

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(message)
        print(f"Message written to {args.output}", file=sys.stderr)
    else:
        print(message)


if __name__ == "__main__":
    main()
