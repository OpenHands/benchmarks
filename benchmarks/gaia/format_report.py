#!/usr/bin/env python3
"""
GAIA Report Formatter

Generates a unified markdown notification message from GAIA evaluation results.
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


def compute_gaia_metrics(output_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute GAIA metrics from output.jsonl data."""
    total = len(output_data)
    success = 0
    errors = 0

    for item in output_data:
        test_result = item.get("test_result", {})
        if test_result.get("score") is True:
            success += 1
        # Count errors if test_result is empty or has error flag
        if not test_result or test_result.get("error"):
            errors += 1

    success_rate = (success / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "success": success,
        "errors": errors,
        "success_rate": success_rate,
    }


def format_gaia_report(
    metrics: dict[str, Any],
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
    Format GAIA evaluation results as a markdown notification.

    Args:
        metrics: Computed metrics dictionary
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
    # Extract GAIA metrics
    total = metrics.get("total", 0)
    success = metrics.get("success", 0)
    success_rate_val = metrics.get("success_rate", 0.0)
    errors = metrics.get("errors", 0)

    # Format success rate
    success_rate = "N/A"
    if total > 0:
        success_rate = f"{success}/{total} ({success_rate_val:.1f}%)"

    # Build markdown message
    lines = [
        "## 🎉 GAIA Evaluation Complete",
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
            f"- **Total instances:** {total}",
            f"- **Successful:** {success}",
            f"- **Errors:** {errors}",
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


def format_gaia_failure(
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
    Format GAIA evaluation failure notification.

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
        "## ❌ GAIA Evaluation Failed",
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
        description="Format GAIA evaluation results for notifications"
    )
    parser.add_argument(
        "output_jsonl",
        help="Path to output.jsonl from evaluation",
    )
    parser.add_argument(
        "report_json",
        nargs="?",
        help="Path to report.json (optional)",
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

    # Load output.jsonl
    try:
        output_data = load_jsonl(args.output_jsonl)
    except Exception as e:
        print(f"Error loading output.jsonl: {e}", file=sys.stderr)
        sys.exit(1)

    # Compute metrics from output.jsonl
    metrics = compute_gaia_metrics(output_data)

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
    dataset = os.environ.get("DATASET", "gaia")
    dataset_split = os.environ.get("DATASET_SPLIT", "validation")
    commit = os.environ.get("COMMIT", "unknown")
    timestamp = os.environ.get("TIMESTAMP", "unknown")

    # Optional variables
    trigger_reason = os.environ.get("TRIGGER_REASON")
    tar_url = os.environ.get("TAR_URL")

    # Format the message
    message = format_gaia_report(
        metrics=metrics,
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
