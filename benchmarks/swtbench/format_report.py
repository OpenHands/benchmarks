#!/usr/bin/env python3
"""
SWT-Bench Report Formatter

Generates a unified markdown notification message from SWT-Bench evaluation results.
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


def compute_swtbench_metrics(
    output_data: list[dict[str, Any]], error_data: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute SWT-Bench metrics from output.jsonl and output_errors.jsonl data."""
    # Count successful instances (those with valid git patches)
    success = 0
    for item in output_data:
        test_result = item.get("test_result", {})
        git_patch = test_result.get("git_patch", "")
        # Consider successful if a non-empty patch was generated
        if git_patch and git_patch.strip():
            success += 1

    # Total is successful + failed instances
    total = len(output_data) + len(error_data)
    errors = len(error_data)
    success_rate = (success / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "success": success,
        "errors": errors,
        "success_rate": success_rate,
    }


def format_swtbench_notification(
    output_path: Path,
    report_path: Path,
    env_vars: dict[str, str],
) -> str:
    """Format SWT-Bench results as markdown notification (aligned with SWE-bench style)."""
    # Load data
    output_data = load_jsonl(str(output_path))
    error_path = output_path.parent / "output_errors.jsonl"
    error_data = load_jsonl(str(error_path)) if error_path.exists() else []

    # Load report if available
    report_data = load_json(str(report_path)) if report_path.exists() else {}

    # Compute metrics
    metrics = compute_swtbench_metrics(output_data, error_data)

    # Extract environment variables
    model_name = env_vars.get("MODEL_NAME", "Unknown")
    eval_limit = env_vars.get("EVAL_LIMIT", "N/A")
    dataset = env_vars.get("DATASET", "princeton-nlp/SWE-bench_Verified")
    dataset_split = env_vars.get("DATASET_SPLIT", "test")
    eval_name = env_vars.get("UNIQUE_EVAL_NAME", "unknown-eval")
    sdk_commit = env_vars.get("COMMIT", "Unknown")[:7]
    timestamp = env_vars.get("TIMESTAMP", "").strip() or env_vars.get(
        "UPLOAD_TIMESTAMP", "unknown time"
    )
    trigger_reason = env_vars.get("TRIGGER_REASON", "").strip()
    tar_url = env_vars.get("TAR_URL", "").strip()

    # Build markdown message
    lines = [
        "## 🎉 SWT-Bench Evaluation Complete",
        "",
        f"**Evaluation:** `{eval_name}`",
        f"**Model:** `{model_name}`",
        f"**Dataset:** `{dataset}` (`{dataset_split}`)",
        f"**Commit:** `{sdk_commit}`",
        f"**Timestamp:** {timestamp}",
    ]

    if trigger_reason:
        lines.append(f"**Reason:** {trigger_reason}")

    lines.extend(
        [
            "",
            "### 📊 Results",
            f"- **Instances:** {metrics['total']} (limit: {eval_limit})",
            f"- **Success rate:** {metrics['success_rate']:.1f}% ({metrics['success']}/{metrics['total']})",
            f"- **Errors:** {metrics['errors']}",
        ]
    )

    # Add report metrics if available
    if report_data:
        lines.append(f"- **Resolved:** {report_data.get('resolved', 0)}")
        lines.append(f"- **Unresolved:** {report_data.get('unresolved', 0)}")

    if tar_url:
        lines.extend(
            [
                "",
                "### 🔗 Links",
                f"[Full Archive]({tar_url})",
            ]
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Format SWT-Bench evaluation results for notifications"
    )
    parser.add_argument("output_jsonl", type=str, help="Path to output.jsonl")
    parser.add_argument("report_json", type=str, help="Path to report.json")
    parser.add_argument(
        "--env-file",
        type=str,
        help="Path to environment file with KEY=VALUE pairs",
        default=None,
    )

    args = parser.parse_args()

    output_path = Path(args.output_jsonl)
    report_path = Path(args.report_json)

    if not output_path.exists():
        print(f"Error: Output file not found: {output_path}", file=sys.stderr)
        return 1

    # Load environment variables
    env_vars = dict(os.environ)
    if args.env_file and Path(args.env_file).exists():
        with open(args.env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    env_vars[key] = value

    # Format and print notification
    message = format_swtbench_notification(output_path, report_path, env_vars)
    print(message)

    return 0


if __name__ == "__main__":
    sys.exit(main())
