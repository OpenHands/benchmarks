#!/usr/bin/env python3
"""
Hybrid-Gym dep_search evaluation script.

Reads output.jsonl from run_infer.py, loads the ground truth dataset,
and evaluates whether the agent correctly annotated all dependencies
with comments. Computes precision, recall, F1 and full success rate.

Usage:
    uv run hybridgym-depsearch-eval output.jsonl --run-id my_run
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from benchmarks.utils.laminar import LaminarService
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Patch parsing
# ---------------------------------------------------------------------------


def parse_patch_with_details(patch_text: str) -> list[dict]:
    """Parse a git patch, returning per-file added lines with line numbers and hunks."""
    if not patch_text:
        return []

    lines = patch_text.split("\n")
    files: list[dict] = []
    current_file = None
    added_lines: list[dict] = []
    hunks: list[tuple] = []
    file_pattern = re.compile(r"^diff --git a/(.+) b/(.+)$")
    hunk_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    current_line = 0

    for line in lines:
        if line.startswith("diff --git"):
            if current_file is not None:
                files.append(
                    {
                        "filename": current_file,
                        "added_lines": added_lines,
                        "hunks": hunks,
                    }
                )
            current_file = None
            added_lines = []
            hunks = []
            current_line = 0
            match = file_pattern.match(line)
            if match:
                current_file = match.group(2)
        elif line.startswith("@@"):
            match = hunk_pattern.match(line)
            if match:
                old_start = int(match.group(1))
                old_count = int(match.group(2)) if match.group(2) else 1
                new_start = int(match.group(3))
                new_count = int(match.group(4)) if match.group(4) else 1
                hunks.append((old_start, old_count, new_start, new_count))
                current_line = new_start
        elif line.startswith("+") and not line.startswith("+++"):
            if current_file is not None:
                added_lines.append({"line_number": current_line, "content": line[1:]})
            current_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            pass  # deleted lines don't advance new-file counter
        elif current_file is not None and not line.startswith(
            ("diff", "@@", "index", "---", "+++")
        ):
            current_line += 1

    if current_file is not None:
        files.append(
            {"filename": current_file, "added_lines": added_lines, "hunks": hunks}
        )
    return files


def compute_line_offset(hunks: list, original_line: int) -> int:
    """Compute how much an original line shifted in the new file due to hunks."""
    if not hunks:
        return 0
    cumulative_offset = 0
    for old_start, old_count, new_start, new_count in sorted(hunks, key=lambda h: h[0]):
        if original_line < old_start:
            return cumulative_offset
        old_end = old_start + old_count
        if original_line < old_end:
            return cumulative_offset + (new_count - old_count)
        cumulative_offset = (new_start + new_count) - (old_start + old_count)
    return cumulative_offset


def is_lines_comment_only(lines: list[str]) -> bool:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            return False
    return True


def check_comment_content(line: str, target_func_name: str) -> bool:
    """Check if a comment mentions the target function as a caller."""
    stripped = line.strip().lower()
    if not stripped.startswith("#"):
        return False
    content = stripped[1:].strip().rstrip(".,;:!")
    target_lower = target_func_name.lower()
    if target_lower in content and (
        "called by" in content or "used by" in content or "dependency" in content
    ):
        return True
    acceptable = [
        f"this function/class is called by the {target_lower} function",
        f"this function is called by the {target_lower} function",
        f"this class is called by the {target_lower} function",
    ]
    return content in acceptable


# ---------------------------------------------------------------------------
# Instance evaluation
# ---------------------------------------------------------------------------


def evaluate_instance(instance_data: dict, output_data: dict) -> dict:
    target_func_name = instance_data["target_function_name"]
    dependencies = instance_data["dependencies"]
    num_deps = len(dependencies)

    result = {
        "instance_id": output_data.get("instance_id"),
        "target_function": target_func_name,
        "num_dependencies": num_deps,
        "true_positives": 0,
        "false_negatives": 0,
        "false_positives": 0,
        "duplicates": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "full_success": False,
        "comments_only": False,
    }

    git_patch = output_data.get("test_result", {}).get("git_patch", "")
    if not git_patch:
        result["false_negatives"] = num_deps
        return result

    patch_details = parse_patch_with_details(git_patch)

    all_content = []
    for pf in patch_details:
        all_content.extend([line["content"] for line in pf["added_lines"]])
    result["comments_only"] = is_lines_comment_only(all_content)

    # Build expected-dependency map: (file, original_1idx_line) -> dep
    expected_deps: dict[tuple, dict] = {}
    for dep in dependencies:
        expected_line = dep.get("decorator_line", dep["line_start"]) + 1
        expected_deps[(dep["file_path"], expected_line)] = dep

    found_deps: set[str] = set()
    dep_counts: dict[str, int] = defaultdict(int)
    all_comments: list[dict] = []

    for file_info in patch_details:
        filename = file_info["filename"]
        hunks = file_info.get("hunks", [])
        for added in file_info["added_lines"]:
            if not check_comment_content(added["content"], target_func_name):
                continue
            line_num = added["line_number"]
            matched = False
            for (dep_file, orig_line), dep in expected_deps.items():
                if dep_file != filename:
                    continue
                adjusted = orig_line + compute_line_offset(hunks, orig_line)
                if abs(line_num - adjusted) <= 5:
                    dep_counts[dep["name"]] += 1
                    status = "duplicate" if dep["name"] in found_deps else "correct"
                    if status == "correct":
                        found_deps.add(dep["name"])
                    all_comments.append({"status": status})
                    matched = True
                    break
            if not matched:
                all_comments.append({"status": "false_positive"})

    tp = len(found_deps)
    fp = sum(1 for c in all_comments if c["status"] == "false_positive")
    dups = sum(1 for c in all_comments if c["status"] == "duplicate")

    total_pos = tp + fp + dups
    precision = tp / total_pos if total_pos > 0 else 0.0
    recall = tp / num_deps if num_deps > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    result.update(
        {
            "true_positives": tp,
            "false_negatives": num_deps - tp,
            "false_positives": fp,
            "duplicates": dups,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "full_success": tp == num_deps
            and fp == 0
            and dups == 0
            and result["comments_only"],
        }
    )
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def generate_report(
    input_file: str,
    output_file: str,
    dataset_name: str,
    split: str,
) -> None:
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
            if eval_result["full_success"]:
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
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    total = len(resolved_ids) + len(unresolved_ids)
    avg_p = sum(r["precision"] for r in results) / max(total, 1)
    avg_r = sum(r["recall"] for r in results) / max(total, 1)
    avg_f1 = sum(r["f1"] for r in results) / max(total, 1)
    logger.info("=== Evaluation Results ===")
    logger.info("Total evaluated: %d", total)
    logger.info(
        "Full success: %d/%d (%.1f%%)",
        len(resolved_ids),
        total,
        100 * len(resolved_ids) / max(total, 1),
    )
    logger.info("Avg Precision: %.4f  Recall: %.4f  F1: %.4f", avg_p, avg_r, avg_f1)
    logger.info("Empty patches: %d  Errors: %d", len(empty_patch_ids), len(error_ids))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid-Gym dep_search outputs and generate report",
    )
    parser.add_argument("input_file", help="Path to output.jsonl from inference")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument(
        "--dataset",
        default="hybrid-gym/hybrid_gym_dep_search",
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
        generate_report(str(input_path), str(report_path), args.dataset, args.split)
        LaminarService.get().update_evaluation_scores(str(input_path), str(report_path))
        generate_cost_report(str(input_path))
        logger.info("Report saved to: %s", report_path)
        print(json.dumps({"report_json": str(report_path)}))
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
