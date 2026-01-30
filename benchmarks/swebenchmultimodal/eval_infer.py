#!/usr/bin/env python3
"""
SWE-Bench Multimodal Evaluation Script

This script converts OpenHands output.jsonl format to SWE-Bench prediction format
and runs the SWE-Bench Multimodal evaluation.

Usage:
    uv run swebenchmultimodal-eval <path_to_output.jsonl>
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.utils.patch_utils import remove_files_from_patch
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)

# Path to ambiguity annotations relative to this file
ANNOTATIONS_FILE = Path(__file__).parent / "ambiguity_annotations.json"


def load_ambiguity_annotations() -> dict[str, Any]:
    """Load the ambiguity annotations for SWE-bench Multimodal instances."""
    if not ANNOTATIONS_FILE.exists():
        logger.warning(f"Ambiguity annotations file not found: {ANNOTATIONS_FILE}")
        return {}

    with open(ANNOTATIONS_FILE, "r") as f:
        data = json.load(f)

    return data.get("annotations", {})


def calculate_component_scores(
    report_json_path: Path,
) -> dict[str, float]:
    """
    Calculate solveable_accuracy, unsolveable_accuracy, and combined_accuracy.

    Args:
        report_json_path: Path to the report.json file from evaluation

    Returns:
        Dictionary with component scores:
        - solveable_accuracy: % of SOLVEABLE instances resolved
        - unsolveable_accuracy: % of non-SOLVEABLE instances resolved
        - combined_accuracy: % of all instances resolved (standard accuracy)
    """
    # Load annotations
    annotations = load_ambiguity_annotations()
    if not annotations:
        logger.warning("No annotations loaded, cannot calculate component scores")
        return {}

    # Load report.json
    if not report_json_path.exists():
        logger.warning(f"Report file not found: {report_json_path}")
        return {}

    with open(report_json_path, "r") as f:
        report = json.load(f)

    resolved_ids = set(report.get("resolved_ids", []))
    total_instances = report.get("total_instances", 0)

    # Separate instances into solveable and unsolveable
    solveable_ids = set()
    unsolveable_ids = set()

    for instance_id, annotation in annotations.items():
        keywords = annotation.get("keywords", [])
        if "SOLVEABLE" in keywords:
            solveable_ids.add(instance_id)
        else:
            unsolveable_ids.add(instance_id)

    # Calculate metrics
    solveable_resolved = len(resolved_ids & solveable_ids)
    unsolveable_resolved = len(resolved_ids & unsolveable_ids)
    total_resolved = len(resolved_ids)

    solveable_total = len(solveable_ids)
    unsolveable_total = len(unsolveable_ids)

    # Calculate accuracies as percentages
    solveable_accuracy = (
        (solveable_resolved / solveable_total * 100) if solveable_total > 0 else 0.0
    )
    unsolveable_accuracy = (
        (unsolveable_resolved / unsolveable_total * 100)
        if unsolveable_total > 0
        else 0.0
    )
    combined_accuracy = (
        (total_resolved / total_instances * 100) if total_instances > 0 else 0.0
    )

    logger.info("Component scores calculation:")
    logger.info(
        f"  SOLVEABLE: {solveable_resolved}/{solveable_total} = {solveable_accuracy:.1f}%"
    )
    logger.info(
        f"  UNSOLVEABLE: {unsolveable_resolved}/{unsolveable_total} = {unsolveable_accuracy:.1f}%"
    )
    logger.info(
        f"  Combined: {total_resolved}/{total_instances} = {combined_accuracy:.1f}%"
    )

    return {
        "solveable_accuracy": round(solveable_accuracy, 1),
        "unsolveable_accuracy": round(unsolveable_accuracy, 1),
        "combined_accuracy": round(combined_accuracy, 1),
        "solveable_resolved": solveable_resolved,
        "solveable_total": solveable_total,
        "unsolveable_resolved": unsolveable_resolved,
        "unsolveable_total": unsolveable_total,
    }


def update_report_with_component_scores(report_json_path: Path) -> dict[str, float]:
    """
    Calculate component scores and update the report.json with them.

    Args:
        report_json_path: Path to the report.json file

    Returns:
        The component scores dictionary
    """
    scores = calculate_component_scores(report_json_path)

    if not scores:
        return {}

    # Load existing report
    with open(report_json_path, "r") as f:
        report = json.load(f)

    # Add component scores to report
    report["component_scores"] = scores

    # Write updated report
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Updated report.json with component scores: {scores}")

    return scores


def convert_to_swebench_format(
    input_file: str, output_file: str, model_name: str | None = None
) -> None:
    """
    Convert OpenHands output.jsonl to SWE-Bench prediction format.

    OpenHands format:
    {
        "instance_id": "django__django-11333",
        "test_result": {
            "git_patch": "diff --git a/file.py b/file.py\n..."
        },
        "instruction": "...",
        "error": null,
        "history": [...]
    }

    SWE-Bench format:
    {
        "instance_id": "django__django-11333",
        "model_patch": "diff --git a/file.py b/file.py\n...",
        "model_name_or_path": "OpenHands/claude-sonnet-4-5-20250929"
    }

    The model identifier is optional. If provided, the value is formatted as
    "OpenHands/{model_name}" where model_name is extracted from the LLM
    config's `model` field (e.g., "litellm_proxy/claude-sonnet-4-5-20250929"
    becomes "claude-sonnet-4-5-20250929"). If not provided, just "OpenHands"
    is used.
    """
    logger.info(f"Converting {input_file} to SWE-Bench format: {output_file}")

    converted_count = 0
    error_count = 0

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                # Extract required fields
                instance_id = data.get("instance_id")
                if not instance_id:
                    logger.warning(f"Line {line_num}: Missing instance_id")
                    error_count += 1
                    continue

                # Extract git_patch from test_result
                test_result = data.get("test_result", {})
                git_patch = test_result.get("git_patch", "")

                if not git_patch:
                    logger.warning(
                        f"Line {line_num}: Missing or empty git_patch for {instance_id}"
                    )
                    # Still create entry with empty patch
                    git_patch = ""

                # postprocess git_patch
                setup_files = ["pyproject.toml", "tox.ini", "setup.py"]
                git_patch = remove_files_from_patch(git_patch, setup_files)

                # Create SWE-Bench format entry
                # Format model_name_or_path as "OpenHands" or "OpenHands/{model_name}"
                if model_name:
                    extracted_model_name = model_name.rsplit("/", 1)[-1]
                    model_name_or_path = f"OpenHands/{extracted_model_name}"
                else:
                    model_name_or_path = "OpenHands"
                swebench_entry = {
                    "instance_id": instance_id,
                    "model_patch": git_patch,
                    "model_name_or_path": model_name_or_path,
                }

                # Write to output file
                outfile.write(json.dumps(swebench_entry) + "\n")
                converted_count += 1

            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON - {e}")
                error_count += 1
            except Exception as e:
                logger.error(f"Line {line_num}: Unexpected error - {e}")
                error_count += 1

    logger.info(
        f"Conversion complete: {converted_count} entries converted, "
        f"{error_count} errors"
    )


def find_report_json(predictions_dir: Path, run_id: str) -> Path | None:
    """
    Find the report.json file generated by SWE-Bench evaluation.

    The report is typically in logs/run_evaluation/{run_id}/{model}/report.json
    or directly in the predictions directory.
    """
    # First try direct location
    direct_report = predictions_dir / "report.json"
    if direct_report.exists():
        return direct_report

    # Try logs directory structure
    logs_dir = predictions_dir / "logs" / "run_evaluation" / run_id
    if logs_dir.exists():
        # Find report.json recursively
        for report_path in logs_dir.rglob("report.json"):
            return report_path

    # Also check for output.report.json pattern
    output_report = predictions_dir / "output.report.json"
    if output_report.exists():
        return output_report

    return None


def run_swebench_multimodal_evaluation(
    predictions_file: str,
    dataset: str = "princeton-nlp/SWE-bench_Multimodal",
    split: str = "dev",
    workers: str = "12",
    run_id: str | None = None,
) -> Path | None:
    """
    Run SWE-Bench Multimodal evaluation on the predictions file.

    Args:
        predictions_file: Path to the SWE-Bench format predictions file
        dataset: SWE-Bench dataset to evaluate against
        split: Dataset split to use (default: dev)
        workers: Number of workers to use for evaluation
        run_id: Optional run ID for the evaluation

    Returns:
        Path to the generated report.json file, or None if not found
    """
    logger.info(f"Running SWE-Bench Multimodal evaluation on {predictions_file}")

    try:
        # Get the directory of the predictions file
        predictions_path = Path(predictions_file)
        predictions_dir = predictions_path.parent
        predictions_filename = predictions_path.name

        # Generate run_id if not provided
        if run_id is None:
            run_id = f"eval_{predictions_path.stem}"

        # Run SWE-Bench Multimodal evaluation using UV environment
        # The key difference from regular SWE-Bench is the --modal true flag
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset,
            "--split",
            split,
            "--predictions_path",
            predictions_filename,
            "--max_workers",
            str(workers),
            "--modal",
            "true",
            "--run_id",
            run_id,
        ]

        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Working directory: {predictions_dir}")
        logger.info("SWE-Bench Multimodal evaluation output:")
        print("-" * 80)

        # Stream output directly to console, running from predictions file directory
        result = subprocess.run(cmd, text=True, cwd=predictions_dir)

        print("-" * 80)
        if result.returncode == 0:
            logger.info("SWE-Bench Multimodal evaluation completed successfully")
        else:
            logger.error(
                f"SWE-Bench Multimodal evaluation failed with return code {result.returncode}"
            )
            raise subprocess.CalledProcessError(result.returncode, cmd)

        # Find and return the report.json path
        report_path = find_report_json(predictions_dir, run_id)
        if report_path:
            logger.info(f"Found report.json at: {report_path}")
        else:
            logger.warning("Could not find report.json after evaluation")

        return report_path

    except FileNotFoundError:
        logger.error(
            "SWE-Bench evaluation command not found. "
            "Make sure SWE-Bench is properly installed."
        )
        raise
    except Exception as e:
        logger.error(f"Error running SWE-Bench Multimodal evaluation: {e}")
        raise


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert OpenHands output to SWE-Bench format and run multimodal evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run swebenchmultimodal-eval output.jsonl
    uv run swebenchmultimodal-eval /path/to/output.jsonl --dataset princeton-nlp/SWE-bench_Multimodal
    uv run swebenchmultimodal-eval output.jsonl --model-name "MyModel-v1.0"
        """,
    )

    parser.add_argument("input_file", help="Path to the OpenHands output.jsonl file")

    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Multimodal",
        help="SWE-Bench dataset to evaluate against "
        "(default: princeton-nlp/SWE-bench_Multimodal)",
    )

    parser.add_argument(
        "--split",
        default="dev",
        help="Dataset split to use (default: dev)",
    )

    parser.add_argument(
        "--output-file",
        help="Output file for SWE-Bench format "
        "(default: input_file with .swebench.jsonl extension)",
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only convert format, skip running evaluation",
    )

    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Optional model identifier. If provided, model_name_or_path will be "
            "'OpenHands/{model_name}' (e.g., litellm_proxy/claude-sonnet-4-5-20250929 "
            "becomes 'OpenHands/claude-sonnet-4-5-20250929'). If not provided, "
            "just 'OpenHands' is used."
        ),
    )

    parser.add_argument(
        "--workers",
        default="12",
        help="Number of workers to use when evaluating",
    )

    parser.add_argument(
        "--run-id",
        help="Run ID for the evaluation (default: eval_<output_filename>)",
    )

    args = parser.parse_args()

    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        sys.exit(1)

    if not input_file.suffix == ".jsonl":
        logger.warning(f"Input file does not have .jsonl extension: {input_file}")

    # Determine output file
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = input_file.with_suffix(".swebench.jsonl")

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Model name: {args.model_name}")

    try:
        # Convert format
        convert_to_swebench_format(str(input_file), str(output_file), args.model_name)

        report_path = None
        if not args.skip_evaluation:
            # Run multimodal evaluation
            report_path = run_swebench_multimodal_evaluation(
                str(output_file), args.dataset, args.split, args.workers, args.run_id
            )

            # Calculate component scores if we have a report
            if report_path:
                logger.info(
                    "Calculating component scores (solveable/unsolveable accuracy)..."
                )
                component_scores = update_report_with_component_scores(report_path)
                if component_scores:
                    logger.info("=" * 60)
                    logger.info("COMPONENT SCORES SUMMARY")
                    logger.info("=" * 60)
                    logger.info(
                        f"  Solveable Accuracy:   {component_scores['solveable_accuracy']:.1f}% "
                        f"({component_scores['solveable_resolved']}/{component_scores['solveable_total']})"
                    )
                    logger.info(
                        f"  Unsolveable Accuracy: {component_scores['unsolveable_accuracy']:.1f}% "
                        f"({component_scores['unsolveable_resolved']}/{component_scores['unsolveable_total']})"
                    )
                    logger.info(
                        f"  Combined Accuracy:    {component_scores['combined_accuracy']:.1f}%"
                    )
                    logger.info("=" * 60)

        # Generate cost report as final step
        generate_cost_report(str(input_file))

        logger.info("Script completed successfully!")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
