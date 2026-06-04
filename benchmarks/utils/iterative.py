"""
Iterative mode utilities for evaluation.

This module contains utilities for implementing iterative mode evaluation,
using SDK critics to determine if an instance succeeded.
"""

import json
import os
from dataclasses import dataclass
from typing import Set

from benchmarks.utils.critics import CriticBase, evaluate_output
from benchmarks.utils.models import EvalInstanceID, EvalOutput
from openhands.sdk import get_logger


logger = get_logger(__name__)


@dataclass
class _AggregatedEntry:
    """Internal representation of a per-instance candidate during aggregation.

    Holds either a fully-parsed ``EvalOutput`` (preferred) or, when pydantic
    validation fails for a line we still want to preserve, the raw JSONL line
    plus enough metadata to rank and filter it.
    """

    instance_id: EvalInstanceID
    rank: int
    error: bool
    output: EvalOutput | None
    raw_line: str | None

    def beats(self, other: "_AggregatedEntry") -> bool:
        """Return True if this entry should replace ``other`` for its instance.

        Mirrors the pre-existing "highest rank wins" tie-break used by
        ``aggregate_results`` before this refactor, and centralises it so both
        the parseable and raw-line fallback paths use identical semantics.
        """
        return self.rank > other.rank

    def serialize(self) -> str:
        """Return the JSONL representation to write to the final output file."""
        if self.output is not None:
            return self.output.model_dump_json() + "\n"
        # Fallback: write the original line verbatim (already includes content
        # that may not round-trip cleanly through the current EvalOutput model,
        # e.g. tool kinds from plugins not registered in this process).
        assert self.raw_line is not None
        return self.raw_line if self.raw_line.endswith("\n") else self.raw_line + "\n"


def _get_output_rank(critic: CriticBase, output: EvalOutput) -> int:
    """
    Get the rank of an output for aggregation purposes.
    Higher rank is better.

    Ranks:
    - 2: critic-successful (best)
    - 1: non-error/critic-fail
    - 0: error (worst)
    """
    if output.error:
        return 0
    if evaluate_output(critic, output):
        return 2
    return 1


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

    if not os.path.exists(output_file):
        logger.warning(f"Output file {output_file} does not exist")
        return failed_instances

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    output = EvalOutput.model_validate(data)

                    # Evaluate using the critic
                    if not evaluate_output(critic, output):
                        failed_instances.add(output.instance_id)

                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Invalid JSON on line {line_num} in {output_file}: {e}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error processing line {line_num} in {output_file}: {e}"
                    )

    except Exception as e:
        logger.error(f"Error reading output file {output_file}: {e}")

    logger.info(f"Found {len(failed_instances)} failed instances in {output_file}")
    return failed_instances


def aggregate_results(
    output_dir: str,
    n_critic_runs: int,
    critic: "CriticBase",
    final_output_file: str = "output.jsonl",
) -> None:
    """
    Aggregate results from multiple attempts into a final output file.

    Works backwards from the last attempt to the first, using the most recent
    successful attempt for each instance.

    Args:
        output_dir: Directory containing attempt files
        n_critic_runs: Number of critic evaluation runs
        critic: Critic instance to use for evaluation
        final_output_file: Name of the final output file
    """
    logger.info(f"Aggregating results from {n_critic_runs} critic runs")

    # Dictionary to store the best candidate for each instance
    best_results: dict[EvalInstanceID, _AggregatedEntry] = {}
    # Track how many entries fell back to raw-line preservation because
    # full EvalOutput validation failed (e.g. resumed runs where the carried
    # over history references discriminated-union "kind"s not registered in
    # this process). Reported in a single summary log to avoid log spam.
    fallback_count = 0

    # Work backwards from the last attempt to the first
    for attempt in range(n_critic_runs, 0, -1):
        attempt_file = os.path.join(
            output_dir, f"output.critic_attempt_{attempt}.jsonl"
        )

        if not os.path.exists(attempt_file):
            logger.debug(f"Attempt file {attempt_file} does not exist, skipping")
            continue

        logger.info(f"Processing attempt {attempt}: {attempt_file}")

        try:
            with open(attempt_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Invalid JSON on line {line_num} in {attempt_file}: {e}"
                        )
                        continue

                    instance_id = data.get("instance_id")
                    if not instance_id:
                        logger.warning(
                            f"Missing instance_id on line {line_num} in "
                            f"{attempt_file}; skipping"
                        )
                        continue

                    # Prefer full pydantic validation so the critic can rank
                    # parseable rows accurately. If that fails, fall back to a
                    # minimal-parse path that preserves the entry as a raw line
                    # so downstream consumers still see the carried-over data.
                    entry: _AggregatedEntry
                    try:
                        output = EvalOutput.model_validate(data)
                        entry = _AggregatedEntry(
                            instance_id=output.instance_id,
                            rank=_get_output_rank(critic, output),
                            error=bool(output.error),
                            output=output,
                            raw_line=None,
                        )
                    except Exception as e:
                        # Most common cause is a discriminated-union kind in
                        # ``history`` that is not registered in this process
                        # (e.g. browser tools after a resume on a pod that does
                        # not load the browser plugin). Conservative rank: 0 if
                        # the row recorded an error, otherwise 1 (non-error,
                        # non-critic-successful). A fully-parseable row for the
                        # same instance from another attempt can still win with
                        # rank 2.
                        has_error = bool(data.get("error"))
                        fallback_count += 1
                        # Log only the exception type at debug level — a single
                        # pydantic ``ValidationError`` can carry dozens of
                        # sub-errors each containing the offending ``input_value``
                        # and would blow up log size in high-volume aggregations.
                        logger.debug(
                            "Falling back to raw-line preservation for line "
                            "%d in %s (instance %s): %s",
                            line_num,
                            attempt_file,
                            instance_id,
                            type(e).__name__,
                        )
                        entry = _AggregatedEntry(
                            instance_id=instance_id,
                            rank=0 if has_error else 1,
                            error=has_error,
                            output=None,
                            raw_line=line,
                        )

                    current = best_results.get(instance_id)
                    if current is None or entry.beats(current):
                        best_results[instance_id] = entry

        except Exception as e:
            logger.error(f"Error reading attempt file {attempt_file}: {e}")

    if fallback_count:
        logger.warning(
            "Preserved %d entries via raw-line fallback (pydantic validation "
            "failed, likely due to history kinds not registered in this "
            "process — e.g. plugin tools from a different run).",
            fallback_count,
        )

    # Write the aggregated results
    final_path = os.path.join(output_dir, final_output_file)
    if not best_results:
        logger.warning("No results found to aggregate - creating empty output file")
    logger.info(f"Writing {len(best_results)} aggregated results to {final_path}")

    try:
        successful_count = 0
        with open(final_path, "w", encoding="utf-8") as f:
            for entry in best_results.values():
                if entry.error:  # Skip outputs with errors
                    continue
                f.write(entry.serialize())
                successful_count += 1

        logger.info(
            f"Successfully wrote {successful_count} successful results to {final_path}"
        )

    except Exception as e:
        logger.error(f"Error writing aggregated results to {final_path}: {e}")
        raise
