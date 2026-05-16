#!/usr/bin/env python3
"""
Hybrid-Gym issue_localize evaluation script.

Reads output.jsonl from run_infer.py and evaluates whether the agent
correctly localized files related to the issue. Success requires:
  1. At least one gold-patch file was touched by the agent's patch
  2. All changes are comments only (no code modifications)

Usage:
    uv run hybridgym-issuelocalize-eval output.jsonl --run-id my_run
"""

import argparse
import json
import re
import sys
from pathlib import Path

from benchmarks.utils.laminar import LaminarService
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Patch parsing
# ---------------------------------------------------------------------------


def patch2file_paths(patch: str) -> set[str]:
    file_paths: set[str] = set()
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                file_paths.add(parts[2][2:])
    return file_paths


def parse_git_patch(patch_text: str) -> list[dict]:
    lines = patch_text.split("\n")
    files: list[dict] = []
    current_file = None
    added_lines: list[str] = []
    removed_lines: list[str] = []
    is_new_file = False
    file_pattern = re.compile(r"^diff --git a/(.+) b/(.+)$")

    for line in lines:
        if line.startswith("diff --git"):
            if current_file is not None:
                files.append(
                    {
                        "filename": current_file,
                        "added_lines": added_lines,
                        "removed_lines": removed_lines,
                        "is_new_file": is_new_file,
                    }
                )
            current_file = None
            added_lines = []
            removed_lines = []
            is_new_file = False
            match = file_pattern.match(line)
            if match:
                current_file = match.group(2)
        elif current_file is not None and line.startswith("new file mode"):
            is_new_file = True
        elif current_file is not None:
            if line.startswith("+") and not line.startswith("+++"):
                content = line[1:].strip()
                if content:
                    added_lines.append(content)
            elif line.startswith("-") and not line.startswith("---"):
                content = line[1:].strip()
                if content:
                    removed_lines.append(content)

    if current_file is not None:
        files.append(
            {
                "filename": current_file,
                "added_lines": added_lines,
                "removed_lines": removed_lines,
                "is_new_file": is_new_file,
            }
        )
    return files


def check_add_comments_only(patch_dict: dict) -> bool:
    """True if the patch only adds comments and doesn't modify code.

    For new files, always returns True.
    For existing files, allows:
    - Pure comment additions (lines starting with #)
    - Lines that match a removed line with an inline comment appended
    """
    if patch_dict["is_new_file"]:
        return True

    removed_stripped = {line.strip() for line in patch_dict["removed_lines"]}
    matched_removed = set()

    for line in patch_dict["added_lines"]:
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if "#" in stripped:
            code_part = stripped.split("#", 1)[0].strip()
            if code_part in removed_stripped:
                matched_removed.add(code_part)
                continue
        if stripped in removed_stripped:
            matched_removed.add(stripped)
            continue
        return False

    for line in patch_dict["removed_lines"]:
        s = line.strip()
        if s.startswith("#") or not s:
            continue
        if s not in matched_removed:
            return False

    return True


# ---------------------------------------------------------------------------
# Instance evaluation
# ---------------------------------------------------------------------------


def evaluate_instance(output_data: dict) -> dict:
    instance_id = output_data.get("instance_id", "")
    instance = output_data.get("instance", {})
    gold_patch = instance.get("patch", "")
    gen_patch = output_data.get("test_result", {}).get("git_patch", "")

    result = {
        "instance_id": instance_id,
        "localization": False,
        "comments_only": False,
        "success": False,
    }

    if not gen_patch.strip():
        return result

    gt_files = patch2file_paths(gold_patch)
    patch_dicts = parse_git_patch(gen_patch)
    gen_files = {pd["filename"] for pd in patch_dicts}

    localization = len(gt_files & gen_files) > 0
    comments_only = all(check_add_comments_only(pd) for pd in patch_dicts)

    result["localization"] = localization
    result["comments_only"] = comments_only
    result["success"] = localization and comments_only
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def generate_report(input_file: str, output_file: str) -> None:
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    error_ids: list[str] = []
    empty_patch_ids: list[str] = []

    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            iid = data.get("instance_id", "")

            if data.get("error"):
                error_ids.append(iid)
                continue

            gen_patch = data.get("test_result", {}).get("git_patch", "")
            if not gen_patch.strip():
                empty_patch_ids.append(iid)
                continue

            r = evaluate_instance(data)
            if r["success"]:
                resolved_ids.append(iid)
            else:
                unresolved_ids.append(iid)

    submitted_ids = resolved_ids + unresolved_ids + error_ids + empty_patch_ids
    report = {
        "schema_version": 2,
        "total_instances": len(submitted_ids),
        "submitted_instances": len(submitted_ids),
        "resolved_instances": len(resolved_ids),
        "resolved_ids": resolved_ids,
        "unresolved_instances": len(unresolved_ids),
        "unresolved_ids": unresolved_ids,
        "error_instances": len(error_ids),
        "error_ids": error_ids,
        "empty_patch_instances": len(empty_patch_ids),
        "empty_patch_ids": empty_patch_ids,
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    total = len(resolved_ids) + len(unresolved_ids)
    logger.info("=== Evaluation Results ===")
    logger.info("Total evaluated: %d", total)
    logger.info(
        "Resolved: %d/%d (%.1f%%)",
        len(resolved_ids),
        total,
        100 * len(resolved_ids) / max(total, 1),
    )
    logger.info("Empty patches: %d  Errors: %d", len(empty_patch_ids), len(error_ids))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid-Gym issue_localize outputs and generate report",
    )
    parser.add_argument("input_file", help="Path to output.jsonl from inference")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        sys.exit(1)

    report_path = input_path.with_suffix(".report.json")

    try:
        generate_report(str(input_path), str(report_path))
        LaminarService.get().update_evaluation_scores(str(input_path), str(report_path))
        generate_cost_report(str(input_path))
        logger.info("Report saved to: %s", report_path)
        print(json.dumps({"report_json": str(report_path)}))
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
