#!/usr/bin/env python3
"""
SWE-Bench Evaluation Script

This script converts OpenHands output.jsonl format to SWE-Bench prediction format
and runs the SWE-Bench evaluation.

Usage:
    uv run swebench-eval <path_to_output.jsonl>
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.utils.patch_utils import remove_files_from_patch
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger
from pydantic import BaseModel, ConfigDict, Field


logger = get_logger(__name__)


class SwebenchReport(BaseModel):
    """SWE-bench compatible evaluation summary."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model_name_or_path: str | None = None

    total_instances: int = Field(
        ge=0, description="Total number of instances in the benchmark split."
    )
    submitted_instances: int = Field(
        ge=0, description="Number of instances submitted for evaluation."
    )
    completed_instances: int = Field(
        ge=0, description="Number of instances with completed, non-error outputs."
    )
    resolved_instances: int = Field(
        ge=0, description="Number of instances marked as resolved/successful."
    )
    unresolved_instances: int = Field(
        ge=0, description="Number of instances marked as unresolved/failed."
    )
    empty_patch_instances: int = Field(
        ge=0, description="Number of instances producing an empty patch."
    )
    error_instances: int = Field(
        ge=0, description="Number of instances that failed with errors."
    )
    incomplete_instances: int | None = Field(
        default=None,
        ge=0,
        description="Number of instances that did not complete (optional).",
    )

    completed_ids: list[str] = Field(
        default_factory=list, description="Instance IDs with completed outputs."
    )
    incomplete_ids: list[str] = Field(
        default_factory=list,
        description="Instance IDs that did not complete or are missing outputs.",
    )
    submitted_ids: list[str] = Field(
        default_factory=list, description="Instance IDs submitted for evaluation."
    )
    resolved_ids: list[str] = Field(
        default_factory=list, description="Instance IDs marked as resolved."
    )
    unresolved_ids: list[str] = Field(
        default_factory=list, description="Instance IDs marked as unresolved."
    )
    empty_patch_ids: list[str] = Field(
        default_factory=list, description="Instance IDs with empty patches."
    )
    error_ids: list[str] = Field(
        default_factory=list, description="Instance IDs that failed with errors."
    )

    schema_version: int | None = 2
    unstopped_instances: int | None = None
    unstopped_containers: list[str] = Field(default_factory=list)
    unremoved_images: list[str] = Field(default_factory=list)

    @classmethod
    def from_ids(
        cls,
        *,
        total_instances: int,
        completed_ids: Sequence[str],
        resolved_ids: Sequence[str],
        unresolved_ids: Sequence[str],
        empty_patch_ids: Sequence[str] | None = None,
        error_ids: Sequence[str] | None = None,
        submitted_ids: Sequence[str] | None = None,
        incomplete_ids: Sequence[str] | None = None,
        model_name_or_path: str | None = None,
    ) -> "SwebenchReport":
        empty_patch_ids_list = list(empty_patch_ids or [])
        error_ids_list = list(error_ids or [])
        completed_ids_list = list(completed_ids)
        resolved_ids_list = list(resolved_ids)
        unresolved_ids_list = list(unresolved_ids)
        submitted_ids_list = (
            list(submitted_ids)
            if submitted_ids is not None
            else list(completed_ids_list)
        )
        incomplete_ids_list = list(incomplete_ids or [])

        return cls(
            model_name_or_path=model_name_or_path,
            total_instances=total_instances,
            submitted_instances=len(submitted_ids_list),
            completed_instances=len(completed_ids_list),
            resolved_instances=len(resolved_ids_list),
            unresolved_instances=len(unresolved_ids_list),
            empty_patch_instances=len(empty_patch_ids_list),
            error_instances=len(error_ids_list),
            incomplete_instances=(
                len(incomplete_ids_list) if incomplete_ids_list else None
            ),
            completed_ids=completed_ids_list,
            incomplete_ids=incomplete_ids_list,
            submitted_ids=submitted_ids_list,
            resolved_ids=resolved_ids_list,
            unresolved_ids=unresolved_ids_list,
            empty_patch_ids=empty_patch_ids_list,
            error_ids=error_ids_list,
        )

    @classmethod
    def from_swtbench_report(cls, report: Mapping[str, Any]) -> "SwebenchReport":
        completed_ids = list(report.get("completed_ids", []))
        resolved_ids = list(report.get("resolved_ids", []))
        unresolved_ids = list(report.get("unresolved_ids", []))
        error_ids = list(report.get("error_ids", []))
        total_instances = int(
            report.get(
                "total_instances",
                len(completed_ids) + len(unresolved_ids) + len(error_ids),
            )
        )

        return cls.from_ids(
            total_instances=total_instances,
            completed_ids=completed_ids,
            resolved_ids=resolved_ids,
            unresolved_ids=unresolved_ids,
            error_ids=error_ids,
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.write_text(
            self.model_dump_json(indent=4, by_alias=True, exclude_none=True)
        )


def convert_to_swebench_format(
    input_file: str, output_file: str, model_name: str = "OpenHands"
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
        "model_name_or_path": "OpenHands"
    }
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
                swebench_entry = {
                    "instance_id": instance_id,
                    "model_patch": git_patch,
                    "model_name_or_path": model_name,
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

    if converted_count == 0:
        raise ValueError("No valid entries were converted")


def run_swebench_evaluation(
    predictions_file: str,
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    workers: str = "12",
) -> None:
    """
    Run SWE-Bench evaluation on the predictions file.

    Args:
        predictions_file: Path to the SWE-Bench format predictions file
        dataset: SWE-Bench dataset to evaluate against
        workers: Number of workers to use for evaluation
    """
    logger.info(f"Running SWE-Bench evaluation on {predictions_file}")

    try:
        # Get the directory of the predictions file
        predictions_path = Path(predictions_file)
        predictions_dir = predictions_path.parent
        predictions_filename = predictions_path.name

        # Run SWE-Bench evaluation using global python (not UV environment)
        # since swebench is installed globally
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset,
            "--predictions_path",
            predictions_filename,
            "--max_workers",
            str(workers),
            "--run_id",
            f"eval_{predictions_path.stem}",
        ]

        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Working directory: {predictions_dir}")
        logger.info("SWE-Bench evaluation output:")
        print("-" * 80)

        # Stream output directly to console, running from predictions file directory
        result = subprocess.run(cmd, text=True, cwd=predictions_dir)

        print("-" * 80)
        if result.returncode == 0:
            logger.info("SWE-Bench evaluation completed successfully")
        else:
            logger.error(
                f"SWE-Bench evaluation failed with return code {result.returncode}"
            )
            raise subprocess.CalledProcessError(result.returncode, cmd)

    except FileNotFoundError:
        logger.error(
            "SWE-Bench evaluation command not found. "
            "Make sure SWE-Bench is properly installed."
        )
        raise
    except Exception as e:
        logger.error(f"Error running SWE-Bench evaluation: {e}")
        raise


def validate_swebench_report(report_path: Path) -> None:
    with report_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    SwebenchReport.model_validate(payload)


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert OpenHands output to SWE-Bench format and run evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run swebench-eval output.jsonl
    uv run swebench-eval /path/to/output.jsonl --dataset princeton-nlp/SWE-bench_Lite
    uv run swebench-eval output.jsonl --model-name "MyModel-v1.0"
        """,
    )

    parser.add_argument("input_file", help="Path to the OpenHands output.jsonl file")

    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Verified",
        help="SWE-Bench dataset to evaluate against "
        "(default: princeton-nlp/SWE-bench_Verified)",
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
        default="openhands",
        help="Model name to use in the model_name_or_path field (default: openhands)",
    )

    parser.add_argument(
        "--workers",
        default="12",
        help="Number of workers to use when evaluating",
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
    logger.info(f"Model name: {args.model_name}")

    try:
        # Convert format
        convert_to_swebench_format(str(input_file), str(output_file), args.model_name)

        if not args.skip_evaluation:
            # Run evaluation
            run_swebench_evaluation(str(output_file), args.dataset, args.workers)

            # Move report file to input file directory with .report.json extension
            # SWE-Bench creates: {model_name.replace("/", "__")}.eval_{output_file.stem}.json
            report_filename = (
                f"{args.model_name.replace('/', '__')}.eval_{output_file.stem}.json"
            )
            report_path = output_file.parent / report_filename
            dest_report_path = input_file.with_suffix(".report.json")

            validate_swebench_report(report_path)
            shutil.move(str(report_path), str(dest_report_path))
            logger.info(f"Moved report file to: {dest_report_path}")

        # Generate cost report as final step
        generate_cost_report(str(input_file))

        logger.info("Script completed successfully!")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
