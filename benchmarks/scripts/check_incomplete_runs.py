#!/usr/bin/env python3
"""Script to check evaluation runs for completeness.

This script analyzes evaluation output directories to detect incomplete runs
where some instances failed to produce any output due to infrastructure issues.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Set

from openhands.sdk import get_logger


logger = get_logger(__name__)


def get_completed_instances(output_file: Path) -> Set[str]:
    """Get the set of instance IDs that completed.

    Args:
        output_file: Path to the output JSONL file

    Returns:
        Set of completed instance IDs
    """
    if not output_file.exists():
        return set()

    completed = set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    instance_id = data.get("instance_id")
                    if instance_id:
                        completed.add(instance_id)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse line {line_num} in {output_file}: {e}"
                    )
    except Exception as e:
        logger.error(f"Failed to read output file {output_file}: {e}")

    return completed


def get_expected_count_from_metadata(metadata_file: Path) -> int | None:
    """Get the expected number of instances from metadata.

    Args:
        metadata_file: Path to the metadata.json file

    Returns:
        Expected instance count or None if not available
    """
    if not metadata_file.exists():
        return None

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Try to get eval_limit
        eval_limit = metadata.get("eval_limit")
        if eval_limit:
            return eval_limit

        return None
    except Exception as e:
        logger.error(f"Failed to read metadata: {e}")
        return None


def analyze_evaluation_run(eval_dir: Path, verbose: bool = False) -> dict:
    """Analyze an evaluation run for completeness.

    Args:
        eval_dir: Path to the evaluation output directory
        verbose: Whether to print verbose output

    Returns:
        Dictionary with analysis results
    """
    if not eval_dir.exists():
        logger.error(f"Directory not found: {eval_dir}")
        return {"error": "Directory not found"}

    if not eval_dir.is_dir():
        logger.error(f"Not a directory: {eval_dir}")
        return {"error": "Not a directory"}

    # Get metadata
    metadata_file = eval_dir / "metadata.json"
    expected_count = get_expected_count_from_metadata(metadata_file)

    # Find all output files
    output_files = []
    if (eval_dir / "output.jsonl").exists():
        output_files.append(("output.jsonl", eval_dir / "output.jsonl"))

    for i in range(1, 10):
        attempt_file = eval_dir / f"output.critic_attempt_{i}.jsonl"
        if attempt_file.exists():
            output_files.append((f"output.critic_attempt_{i}.jsonl", attempt_file))

    if not output_files:
        logger.warning(f"No output files found in {eval_dir}")
        return {
            "error": "No output files found",
            "expected_count": expected_count,
        }

    # Analyze each output file
    results = {}
    for name, path in output_files:
        completed = get_completed_instances(path)
        results[name] = {
            "completed_count": len(completed),
            "completed_instances": sorted(completed),
        }

        if verbose:
            logger.info(f"\n{name}:")
            logger.info(f"  Completed: {len(completed)}")

            if expected_count:
                missing_count = expected_count - len(completed)
                if missing_count > 0:
                    logger.warning(
                        f"  Missing: {missing_count} instances "
                        f"(expected {expected_count})"
                    )
                elif missing_count < 0:
                    logger.warning(
                        f"  Extra: {-missing_count} instances "
                        f"(expected {expected_count})"
                    )
                else:
                    logger.info(f"  ✓ Complete ({expected_count} instances)")

    # Check final output
    final_output = results.get("output.jsonl")
    if final_output and expected_count:
        completed_count = final_output["completed_count"]
        is_complete = completed_count == expected_count

        if not is_complete:
            logger.error("=" * 80)
            logger.error("INCOMPLETE EVALUATION RUN DETECTED")
            logger.error("=" * 80)
            logger.error(f"Directory: {eval_dir}")
            logger.error(f"Expected instances: {expected_count}")
            logger.error(f"Completed instances: {completed_count}")
            logger.error(f"Missing instances: {expected_count - completed_count}")
            logger.error("=" * 80)
            logger.error("POSSIBLE CAUSES:")
            logger.error("- Infrastructure failures (Modal, Docker, network)")
            logger.error("- Worker process crashes")
            logger.error("- Resource exhaustion (OOM, disk space)")
            logger.error("- Scheduler issues")
            logger.error("=" * 80)
        else:
            logger.info("=" * 80)
            logger.info("✓ EVALUATION RUN IS COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Directory: {eval_dir}")
            logger.info(f"Instances: {completed_count}/{expected_count}")
            logger.info("=" * 80)

    return {
        "expected_count": expected_count,
        "output_files": results,
        "is_complete": (
            final_output["completed_count"] == expected_count
            if final_output and expected_count
            else None
        ),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check evaluation runs for completeness"
    )
    parser.add_argument(
        "eval_dir",
        type=str,
        help="Path to the evaluation output directory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    result = analyze_evaluation_run(eval_dir, verbose=args.verbose)

    if "error" in result:
        sys.exit(1)

    if result.get("is_complete") is False:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
