"""
Iterative mode utilities for evaluation.

This module contains utilities for implementing iterative mode evaluation,
using SDK critics to determine if an instance succeeded.
"""

import os
from pathlib import Path
from typing import Set

from benchmarks.utils.critics import CriticBase
from benchmarks.utils.models import EvalInstanceID
from benchmarks.utils.output_schema import load_output_file
from openhands.sdk import get_logger


logger = get_logger(__name__)


def get_failed_instances(output_file: str, critic: CriticBase) -> Set[EvalInstanceID]:
    """
    Get the set of failed instance IDs from an output file.

    Args:
        output_file: Path to the JSONL output file
        critic: SDK critic to use for evaluation

    Returns:
        Set of instance IDs that failed
    """

    failed_instances: Set[EvalInstanceID] = set()
    _ = critic  # retained for compatibility

    if not os.path.exists(output_file):
        logger.warning(f"Output file {output_file} does not exist")
        return failed_instances

    try:
        outputs = load_output_file(output_file)
        for out in outputs:
            if out.resolved:
                continue
            failed_instances.add(out.instance_id)
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Error reading output file {output_file}: {e}")

    logger.info("Found %d failed instances in %s", len(failed_instances), output_file)
    return failed_instances


def aggregate_results(
    output_dir: str,
    max_attempts: int,
    critic: "CriticBase",
    final_output_file: str = "output.jsonl",
) -> None:
    """
    Aggregate results from multiple attempts into a final output file.

    Works backwards from the last attempt to the first, using the most recent
    successful attempt for each instance.

    Args:
        output_dir: Directory containing attempt files
        max_attempts: Maximum number of attempts
        critic: Critic instance to use for evaluation
        final_output_file: Name of the final output file
    """
    logger.info("Aggregating results from %s attempts", max_attempts)

    final_path = Path(output_dir) / final_output_file
    final_path.unlink(missing_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        attempt_file = Path(output_dir) / f"output.critic_attempt_{attempt}.jsonl"
        if not attempt_file.exists():
            continue
        with open(final_path, "a", encoding="utf-8") as dest, open(
            attempt_file, "r", encoding="utf-8"
        ) as src:
            for line in src:
                dest.write(line)

    logger.info("Wrote consolidated attempts to %s", final_path)
