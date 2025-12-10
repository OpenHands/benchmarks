"""Utility for detecting and reporting incomplete evaluation runs."""

import json
from pathlib import Path
from typing import Set

from openhands.sdk import get_logger


logger = get_logger(__name__)


class IncompleteRunDetector:
    """Detects and reports on incomplete evaluation runs."""

    def __init__(self, eval_output_dir: str):
        """Initialize the detector.

        Args:
            eval_output_dir: Path to the evaluation output directory
        """
        self.eval_output_dir = Path(eval_output_dir)

    def get_expected_instances(self) -> Set[str]:
        """Get the set of instance IDs that were expected to be processed.

        Reads from metadata.json to determine which instances should have been evaluated.

        Returns:
            Set of expected instance IDs
        """
        metadata_file = self.eval_output_dir / "metadata.json"
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found: {metadata_file}")
            return set()

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                json.load(f)

            # The metadata doesn't store the list of instances directly,
            # so we need to reconstruct it from the dataset
            # This is a placeholder - the actual implementation would need
            # to reload the dataset with the same parameters
            logger.warning(
                "Expected instances cannot be determined from metadata alone. "
                "Need to reload dataset with original parameters."
            )
            return set()
        except Exception as e:
            logger.error(f"Failed to read metadata: {e}")
            return set()

    def get_completed_instances(self, output_file: str = "output.jsonl") -> Set[str]:
        """Get the set of instance IDs that completed (successfully or with errors).

        Args:
            output_file: Name of the output file to check

        Returns:
            Set of completed instance IDs
        """
        output_path = self.eval_output_dir / output_file
        if not output_path.exists():
            logger.warning(f"Output file not found: {output_path}")
            return set()

        completed = set()
        try:
            with open(output_path, "r", encoding="utf-8") as f:
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
            logger.error(f"Failed to read output file {output_path}: {e}")

        return completed

    def check_completeness(
        self,
        expected_instances: Set[str],
        output_file: str = "output.jsonl",
    ) -> dict:
        """Check if the evaluation run is complete.

        Args:
            expected_instances: Set of instance IDs that were expected
            output_file: Name of the output file to check

        Returns:
            Dictionary with completeness information:
            - is_complete: bool
            - expected_count: int
            - completed_count: int
            - missing_count: int
            - missing_instances: List[str]
        """
        completed = self.get_completed_instances(output_file)

        missing = expected_instances - completed
        extra = completed - expected_instances

        result = {
            "is_complete": len(missing) == 0,
            "expected_count": len(expected_instances),
            "completed_count": len(completed),
            "missing_count": len(missing),
            "missing_instances": sorted(missing),
            "extra_count": len(extra),
            "extra_instances": sorted(extra),
        }

        return result

    def report_incomplete_run(
        self,
        expected_instances: Set[str],
        output_file: str = "output.jsonl",
    ) -> None:
        """Log a warning if the evaluation run is incomplete.

        Args:
            expected_instances: Set of instance IDs that were expected
            output_file: Name of the output file to check
        """
        result = self.check_completeness(expected_instances, output_file)

        if not result["is_complete"]:
            logger.error(
                "=" * 80 + "\n"
                "INCOMPLETE EVALUATION RUN DETECTED\n"
                "=" * 80 + "\n"
                f"Expected instances: {result['expected_count']}\n"
                f"Completed instances: {result['completed_count']}\n"
                f"Missing instances: {result['missing_count']}\n"
                "=" * 80
            )

            if result["missing_instances"]:
                logger.error("Missing instance IDs:")
                for inst_id in result["missing_instances"]:
                    logger.error(f"  - {inst_id}")

            if result["extra_instances"]:
                logger.warning(
                    f"Found {result['extra_count']} unexpected instances "
                    "(not in expected list):"
                )
                for inst_id in result["extra_instances"]:
                    logger.warning(f"  - {inst_id}")

            logger.error(
                "=" * 80 + "\n"
                "POSSIBLE CAUSES:\n"
                "- Infrastructure failures (Modal, Docker, network)\n"
                "- Worker process crashes before returning results\n"
                "- Resource exhaustion (OOM, disk space)\n"
                "- Scheduler issues dropping tasks\n"
                "=" * 80
            )
        else:
            logger.info(
                f"Evaluation run is complete: "
                f"{result['completed_count']}/{result['expected_count']} instances"
            )


def detect_incomplete_runs_in_directory(eval_output_dir: str) -> None:
    """Scan an evaluation output directory and report on incomplete runs.

    This is a utility function for post-hoc analysis of evaluation runs.

    Args:
        eval_output_dir: Path to the evaluation output directory
    """
    detector = IncompleteRunDetector(eval_output_dir)

    # Check all output files (including attempt-specific files)
    output_files = []
    eval_path = Path(eval_output_dir)

    # Main output file
    if (eval_path / "output.jsonl").exists():
        output_files.append("output.jsonl")

    # Attempt-specific files
    for i in range(1, 10):  # Check up to 10 attempts
        attempt_file = f"output.critic_attempt_{i}.jsonl"
        if (eval_path / attempt_file).exists():
            output_files.append(attempt_file)

    if not output_files:
        logger.warning(f"No output files found in {eval_output_dir}")
        return

    logger.info(f"Found {len(output_files)} output files: {output_files}")

    for output_file in output_files:
        logger.info(f"\nAnalyzing {output_file}...")
        completed = detector.get_completed_instances(output_file)
        logger.info(f"  Completed instances: {len(completed)}")

        # List instance IDs
        if completed:
            logger.info("  Instance IDs:")
            for inst_id in sorted(completed):
                logger.info(f"    - {inst_id}")
