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
import subprocess
from pathlib import Path

from benchmarks.multiswebench.scripts.eval.update_multi_swe_bench_config import (
    update_multi_swe_config,
)
from openhands.sdk import get_logger


logger = get_logger(__name__)

# Environment variables for multi-language support
LANGUAGE = os.environ.get("LANGUAGE", "java")


def run_multi_swebench_evaluation(
    predictions_file: str,
    dataset_name: str | None = None,
    split: str | None = None,
    original_file: str | None = None,
) -> dict:
    """
    Run Multi-SWE-Bench evaluation using the predictions file.

    Args:
        predictions_file: Path to the predictions JSON file
        dataset_name: Name of the dataset (e.g., "bytedance-research/Multi-SWE-Bench")
        split: Dataset split (e.g., "test", "train")
        original_file: Path to the original OpenHands output.jsonl file

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
        # Load predictions for basic statistics
        with open(predictions_file, "r") as f:
            predictions = json.load(f)

        total_instances = len(predictions)
        instances_with_patches = sum(
            1 for p in predictions if p.get("model_patch", "").strip()
        )

        logger.info(f"Total instances: {total_instances}")
        logger.info(f"Instances with patches: {instances_with_patches}")

        # Use the same directory as the predictions file for evaluation
        predictions_path = Path(predictions_file)
        work_dir = predictions_path.parent

        # Copy predictions file to work directory with a specific name
        temp_jsonl = work_dir / "predictions.json"
        with open(temp_jsonl, "w") as f:
            json.dump(predictions, f, indent=2)

        # Create config file for Multi-SWE-Bench
        config_file = work_dir / "config.json"

        # Handle dataset path if provided
        dataset_path = str(Path(dataset_name).resolve())

        config_input_file = original_file if original_file else str(temp_jsonl)
        update_multi_swe_config(config_input_file, str(config_file), dataset_path)

        logger.info(f"Generated config file: {config_file}")

        # Run the Multi-SWE-Bench evaluation
        logger.info("Running Multi-SWE-Bench evaluation harness...")

        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "multi_swe_bench.harness.run_evaluation",
            "--config",
            str(config_file.resolve()),
            "--mode",
            "evaluation",
        ]

        logger.info(f"Evaluation command: {' '.join(cmd)}")

        # Run with real-time output streaming
        result = subprocess.run(cmd, cwd=work_dir)

        logger.info(f"Return code: {result.returncode}")

        if result.returncode != 0:
            error_msg = f"Evaluation failed with return code {result.returncode}"
            print(f"ERROR: {error_msg}")
            logger.error(error_msg)
            return {
                "total_instances": total_instances,
                "instances_with_patches": instances_with_patches,
                "patch_rate": instances_with_patches / total_instances
                if total_instances > 0
                else 0,
                "language": LANGUAGE,
                "dataset": dataset_name,
                "split": split,
                "evaluation_status": "error",
                "error": error_msg,
            }

        # Parse evaluation results
        # Look for the report file in the evaluation output directory
        eval_files_dir = work_dir / "eval_files"
        report_files = list(eval_files_dir.glob("**/report.json"))

        if report_files:
            with open(report_files[0], "r") as f:
                eval_results = json.load(f)

            # Extract key metrics from the evaluation results
            results = {
                "total_instances": total_instances,
                "instances_with_patches": instances_with_patches,
                "patch_rate": instances_with_patches / total_instances
                if total_instances > 0
                else 0,
                "language": LANGUAGE,
                "dataset": dataset_name,
                "split": split,
                "evaluation_status": "success",
                "eval_results": eval_results,
            }

            # Add summary metrics if available
            if "summary" in eval_results:
                results.update(eval_results["summary"])

        else:
            # If no report file found, this means no instances were processed
            error_msg = "No evaluation report generated - likely all instances were rejected as 'not registered'"
            print(f"ERROR: {error_msg}")
            logger.error(error_msg)
            return {
                "total_instances": total_instances,
                "instances_with_patches": instances_with_patches,
                "patch_rate": instances_with_patches / total_instances
                if total_instances > 0
                else 0,
                "language": LANGUAGE,
                "dataset": dataset_name,
                "split": split,
                "evaluation_status": "error",
                "error": error_msg,
            }

        logger.info(f"Evaluation results: {results}")
        return results

    except Exception as e:
        error_msg = f"Error running evaluation: {e}"
        print(f"ERROR: {error_msg}")
        logger.error(error_msg)
        return {
            "total_instances": 0,
            "instances_with_patches": 0,
            "patch_rate": 0,
            "language": LANGUAGE,
            "dataset": dataset_name,
            "split": split,
            "evaluation_status": "error",
            "error": str(e),
        }


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

    # Run evaluation if not skipped
    if not args.skip_evaluation:
        results = run_multi_swebench_evaluation(
            output_file, args.dataset, args.split, args.input_file
        )

        # Save results
        results_file = str(Path(output_file).with_suffix(".results.json"))
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
