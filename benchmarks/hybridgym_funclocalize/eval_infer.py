#!/usr/bin/env python3
"""
Hybrid-Gym func_localize evaluation script.

Reads output.jsonl from run_infer.py, loads the ground truth dataset,
and evaluates whether the agent:
  1. Added a docstring within the target function/class line range
  2. Made only comment/docstring changes (no code modifications)

Usage:
    uv run hybridgym-funclocalize-eval output.jsonl --run-id my_run
"""

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset

from benchmarks.utils.laminar import LaminarService
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Patch parsing utilities
# ---------------------------------------------------------------------------


def patch2file_paths(patch: str) -> set[str]:
    """Extract all modified file paths from a git patch."""
    file_paths: set[str] = set()
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                file_paths.add(parts[2][2:])  # strip 'a/' prefix
    return file_paths


def parse_git_patch(patch_text: str) -> list[dict]:
    """Parse a git patch into per-file added/removed lines."""
    lines = patch_text.split("\n")
    files: list[dict] = []
    current_file = None
    added_lines: list[str] = []
    removed_lines: list[str] = []
    file_pattern = re.compile(r"^diff --git a/(.+) b/(.+)$")

    for line in lines:
        if line.startswith("diff --git"):
            if current_file is not None:
                files.append(
                    {
                        "filename": current_file,
                        "added_lines": added_lines,
                        "removed_lines": removed_lines,
                    }
                )
            current_file = None
            added_lines = []
            removed_lines = []
            match = file_pattern.match(line)
            if match:
                current_file = match.group(2)
        elif (
            line.startswith("+")
            and not line.startswith("+++")
            and current_file is not None
        ):
            content = line[1:].strip()
            if content:
                added_lines.append(content)
        elif (
            line.startswith("-")
            and not line.startswith("---")
            and current_file is not None
        ):
            content = line[1:].strip()
            if content:
                removed_lines.append(content)

    if current_file is not None:
        files.append(
            {
                "filename": current_file,
                "added_lines": added_lines,
                "removed_lines": removed_lines,
            }
        )
    return files


def parse_hunk_line_numbers(patch_text: str) -> list[dict]:
    """Parse a git patch and extract line numbers where content was added."""
    lines = patch_text.split("\n")
    files: list[dict] = []
    current_file = None
    added_line_numbers: list[int] = []
    hunk_pattern = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    file_pattern = re.compile(r"^diff --git a/(.+) b/(.+)$")
    current_line = 0

    for line in lines:
        if line.startswith("diff --git"):
            if current_file is not None:
                files.append(
                    {"filename": current_file, "added_line_numbers": added_line_numbers}
                )
            current_file = None
            added_line_numbers = []
            match = file_pattern.match(line)
            if match:
                current_file = match.group(2)
        elif line.startswith("@@"):
            match = hunk_pattern.match(line)
            if match:
                current_line = int(match.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            added_line_numbers.append(current_line)
            current_line += 1
        elif (
            not line.startswith("-")
            and not line.startswith("---")
            and current_file is not None
        ):
            if line.startswith(" ") or (
                line and not line.startswith(("diff", "@@", "index", "---", "+++"))
            ):
                current_line += 1

    if current_file is not None:
        files.append(
            {"filename": current_file, "added_line_numbers": added_line_numbers}
        )
    return files


# ---------------------------------------------------------------------------
# Comment / docstring detection
# ---------------------------------------------------------------------------


def is_lines_comment_only(lines: list[str]) -> bool:
    """Return True if every line is a comment, docstring, or whitespace."""
    if not lines:
        return True
    in_docstring = False
    docstring_char = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if in_docstring:
            if docstring_char and docstring_char in stripped:
                if (
                    stripped.endswith(docstring_char)
                    or stripped.count(docstring_char) % 2 == 1
                ):
                    in_docstring = False
                    docstring_char = None
            continue
        if stripped.startswith("#"):
            continue
        for quote in ['"""', "'''"]:
            if stripped.startswith(quote):
                if stripped.count(quote) >= 2:
                    after_first = stripped[3:]
                    if quote in after_first:
                        break  # single-line docstring
                    else:
                        in_docstring = True
                        docstring_char = quote
                        break
                else:
                    in_docstring = True
                    docstring_char = quote
                    break
        else:
            if not in_docstring:
                return False
    return True


def check_add_comments_only(patch_dict: dict) -> bool:
    """True if the patch only adds comments/docstrings and removes nothing."""
    if patch_dict["removed_lines"]:
        return False
    return is_lines_comment_only(patch_dict["added_lines"])


# ---------------------------------------------------------------------------
# Instance-level evaluation
# ---------------------------------------------------------------------------


def evaluate_instance(instance_data: dict, output_data: dict) -> dict:
    """Evaluate a single instance.

    Success = target_docstring_edited AND comments_only.
    """
    history = output_data.get("history") or []
    result = {
        "instance_id": output_data.get("instance_id"),
        "num_steps": len(history),
        "target_docstring_edited": False,
        "comments_only": False,
        "success": False,
    }

    git_patch = output_data.get("test_result", {}).get("git_patch", "")
    if not git_patch:
        return result

    patch_dicts = parse_git_patch(git_patch)
    line_info = parse_hunk_line_numbers(git_patch)

    # Determine targets
    if "functions" in instance_data and instance_data["functions"]:
        targets = instance_data["functions"]
    else:
        targets = [
            {
                "file_path": instance_data.get("file_path"),
                "module_line_start": instance_data.get("module_line_start"),
                "module_line_end": instance_data.get("module_line_end"),
            }
        ]

    # Check all changes are comments/docstrings only
    all_comments = all(check_add_comments_only(pd) for pd in patch_dicts)
    result["comments_only"] = all_comments

    # Check target function had a docstring added within its line range
    target_edited = False
    for file_info in line_info:
        for target in targets:
            if target.get("file_path") == file_info["filename"]:
                raw_start = target.get("module_line_start")
                raw_end = target.get("module_line_end")
                if raw_start is None or raw_end is None:
                    continue
                start = int(raw_start) + 1  # 0-indexed → 1-indexed
                end = int(raw_end) + 1
                if any(start <= ln <= end for ln in file_info["added_line_numbers"]):
                    target_edited = True
                    break
        if target_edited:
            break

    result["target_docstring_edited"] = target_edited
    result["success"] = target_edited and all_comments
    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    input_file: str,
    output_file: str,
    dataset_name: str,
    split: str,
) -> None:
    """Generate evaluation report from output.jsonl + ground truth."""
    # Load ground truth
    instance_id2data: dict[str, dict] = {}
    if dataset_name.endswith((".jsonl", ".json")):
        with open(dataset_name) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    instance_id2data[data["instance_id"]] = data
    else:
        dataset = load_dataset(dataset_name, split=split)
        for row in dataset:  # type: ignore[union-attr]
            row_dict = dict(row)  # type: ignore[call-overload]
            instance_id2data[row_dict["instance_id"]] = row_dict
    logger.info("Loaded %d ground truth instances", len(instance_id2data))

    # Evaluate
    results: list[dict] = []
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    error_ids: list[str] = []
    empty_patch_ids: list[str] = []

    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            output_data = json.loads(line)
            iid = output_data.get("instance_id", "")

            if output_data.get("error"):
                error_ids.append(iid)
                continue

            git_patch = output_data.get("test_result", {}).get("git_patch", "")
            if not git_patch.strip():
                empty_patch_ids.append(iid)
                continue

            if iid not in instance_id2data:
                logger.warning("instance_id %s not found in ground truth", iid)
                unresolved_ids.append(iid)
                continue

            eval_result = evaluate_instance(instance_id2data[iid], output_data)
            results.append(eval_result)
            if eval_result["success"]:
                resolved_ids.append(iid)
            else:
                unresolved_ids.append(iid)

    submitted_ids = resolved_ids + unresolved_ids + error_ids + empty_patch_ids
    report = {
        "schema_version": 2,
        "total_instances": len(submitted_ids),
        "submitted_instances": len(submitted_ids),
        "submitted_ids": submitted_ids,
        "completed_instances": len(resolved_ids) + len(unresolved_ids),
        "completed_ids": resolved_ids + unresolved_ids,
        "resolved_instances": len(resolved_ids),
        "resolved_ids": resolved_ids,
        "unresolved_instances": len(unresolved_ids),
        "unresolved_ids": unresolved_ids,
        "error_instances": len(error_ids),
        "error_ids": error_ids,
        "empty_patch_instances": len(empty_patch_ids),
        "empty_patch_ids": empty_patch_ids,
        "incomplete_ids": error_ids + empty_patch_ids,
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    total = len(resolved_ids) + len(unresolved_ids)
    target_edited = sum(1 for r in results if r["target_docstring_edited"])
    comments_only = sum(1 for r in results if r["comments_only"])
    logger.info("=== Evaluation Results ===")
    logger.info("Total evaluated: %d", total)
    logger.info(
        "Resolved (success): %d/%d (%.1f%%)",
        len(resolved_ids),
        total,
        100 * len(resolved_ids) / max(total, 1),
    )
    logger.info("Target docstring edited: %d/%d", target_edited, total)
    logger.info("Comments only: %d/%d", comments_only, total)
    logger.info("Empty patches: %d", len(empty_patch_ids))
    logger.info("Errors: %d", len(error_ids))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid-Gym func_localize outputs and generate report",
    )
    parser.add_argument("input_file", help="Path to output.jsonl from inference")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument(
        "--dataset",
        default="hybrid-gym/hybrid_gym_func_localize",
        help="HuggingFace dataset name or local JSONL path for ground truth",
    )
    parser.add_argument("--split", default="train", help="Dataset split")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        sys.exit(1)

    report_path = input_path.with_suffix(".report.json")

    try:
        generate_report(
            str(input_path),
            str(report_path),
            args.dataset,
            args.split,
        )
        LaminarService.get().update_evaluation_scores(str(input_path), str(report_path))
        generate_cost_report(str(input_path))
        logger.info("Report saved to: %s", report_path)
        print(json.dumps({"report_json": str(report_path)}))
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
