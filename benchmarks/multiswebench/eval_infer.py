#!/usr/bin/env python3
"""
Multi-SWE-Bench Evaluation Script

This script converts OpenHands output.jsonl format to Multi-SWE-Bench prediction format
and runs the Multi-SWE-Bench evaluation.

Usage:
    uv run multi-swebench-eval <path_to_output.jsonl>
"""

import argparse
import json
import os
from pathlib import Path

from benchmarks.utils.patch_utils import remove_files_from_patch
from openhands.sdk import get_logger


logger = get_logger(__name__)

# Environment variables for multi-language support
LANGUAGE = os.environ.get("LANGUAGE", "java")


def convert_to_multi_swebench_format(
    input_file: str, output_file: str, model_name: str = "OpenHands"
) -> None:
    """
    Convert OpenHands output.jsonl to Multi-SWE-Bench prediction format.

    OpenHands format:
    {
        "instance_id": "repo__version",
        "test_result": {
            "git_patch": "diff --git a/file.py b/file.py\n..."
        },
        "instruction": "...",
        "error": null,
        "history": [...]
    }

    Multi-SWE-Bench format:
    {
        "instance_id": "repo__version",
        "model_patch": "diff --git a/file.py b/file.py\n...",
        "model_name_or_path": "OpenHands"
    }
    """
    logger.info(f"Converting {input_file} to Multi-SWE-Bench format: {output_file}")

    predictions = []

    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())

            instance_id = data.get("instance_id")
            if not instance_id:
                logger.warning(f"Missing instance_id in line: {line}")
                continue

            # Extract git patch
            git_patch = ""
            if "test_result" in data and data["test_result"]:
                git_patch = data["test_result"].get("git_patch", "")
            elif "git_patch" in data:
                git_patch = data["git_patch"]

            # Remove test files from patch if needed
            if git_patch:
                git_patch = remove_files_from_patch(git_patch, ["test_", "_test"])

            prediction = {
                "instance_id": instance_id,
                "model_patch": git_patch,
                "model_name_or_path": model_name,
            }

            predictions.append(prediction)

    # Write predictions to output file
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)

    logger.info(f"Converted {len(predictions)} predictions to {output_file}")


def run_multi_swebench_evaluation(
    predictions_file: str, dataset_name: str | None = None, split: str | None = None
) -> dict:
    """
    Run Multi-SWE-Bench evaluation using the predictions file.

    Args:
        predictions_file: Path to the predictions JSON file
        dataset_name: Name of the dataset (e.g., "bytedance-research/Multi-SWE-Bench")
        split: Dataset split (e.g., "test", "train")

    Returns:
        Dictionary containing evaluation results
    """
    logger.info(f"Running Multi-SWE-Bench evaluation on {predictions_file}")

    # Default dataset and split if not provided
    if dataset_name is None:
        dataset_name = "bytedance-research/Multi-SWE-Bench"
    if split is None:
        split = "test"

    try:
        # For now, we'll use a simplified evaluation approach
        # In a full implementation, this would call the Multi-SWE-Bench evaluation toolkit

        # Load predictions
        with open(predictions_file, "r") as f:
            predictions = json.load(f)

        # Basic statistics
        total_instances = len(predictions)
        instances_with_patches = sum(
            1 for p in predictions if p.get("model_patch", "").strip()
        )

        results = {
            "total_instances": total_instances,
            "instances_with_patches": instances_with_patches,
            "patch_rate": instances_with_patches / total_instances
            if total_instances > 0
            else 0,
            "language": LANGUAGE,
            "dataset": dataset_name,
            "split": split,
        }

        logger.info(f"Evaluation results: {results}")
        return results

    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        return {"error": str(e)}


def main():
    """Main entry point for Multi-SWE-Bench evaluation."""
    parser = argparse.ArgumentParser(description="Multi-SWE-Bench Evaluation")
    parser.add_argument("input_file", help="Path to OpenHands output.jsonl file")
    parser.add_argument(
        "--output-file",
        help="Path to output predictions file (default: input_file.predictions.json)",
    )
    parser.add_argument(
        "--model-name", default="OpenHands", help="Model name for predictions"
    )
    parser.add_argument(
        "--dataset", default="bytedance-research/Multi-SWE-Bench", help="Dataset name"
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip running evaluation, only convert format",
    )

    args = parser.parse_args()

    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        input_path = Path(args.input_file)
        output_file = str(input_path.with_suffix(".predictions.json"))

    # Convert format
    convert_to_multi_swebench_format(args.input_file, output_file, args.model_name)

    # Run evaluation if not skipped
    if not args.skip_evaluation:
        results = run_multi_swebench_evaluation(output_file, args.dataset, args.split)

        # Save results
        results_file = str(Path(output_file).with_suffix(".results.json"))
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
