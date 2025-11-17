"""
Evaluation orchestrator.
"""

import json
import os
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel, Field
from tqdm import tqdm

from benchmarks.utils.constants import OUTPUT_FILENAME
from benchmarks.utils.critics import get_completed_instances
from benchmarks.utils.iterative import aggregate_results, get_failed_instances
from benchmarks.utils.models import (
    EvalInstance,
    EvalInstanceID,
    EvalMetadata,
    EvalOutput,
)
from openhands.sdk import get_logger
from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)

OnResult = Callable[[EvalInstance, EvalOutput], None]


class Evaluation(ABC, BaseModel):
    """Abstract orchestrator for instance processing (process-based)."""

    metadata: EvalMetadata
    num_workers: int = Field(default=1, ge=1)

    def model_post_init(self, __context) -> None:
        """Save metadata to output directory after initialization."""
        # Ensure output directory exists
        os.makedirs(self.metadata.eval_output_dir, exist_ok=True)

        # Save metadata to JSON file
        metadata_file = os.path.join(self.metadata.eval_output_dir, "metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            f.write(self.metadata.model_dump_json(indent=2))
        logger.info(f"Saved metadata to {metadata_file}")

    @property
    def output_path(self) -> str:
        return os.path.join(self.metadata.eval_output_dir, OUTPUT_FILENAME)

    def _get_completed_instances(self) -> set[EvalInstanceID]:
        """Return the set of completed instance IDs."""
        completed_instances: set[EvalInstanceID] = set()
        if os.path.exists(self.output_path):
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    out = json.loads(line)
                    completed_instances.add(out["instance_id"])
            logger.info(
                f"Found {len(completed_instances)} completed instances "
                f"in {self.output_path}"
            )
        return completed_instances

    @abstractmethod
    def prepare_instances(self) -> List[EvalInstance]:
        """Return the list of instances to evaluate."""
        raise NotImplementedError

    @abstractmethod
    def prepare_workspace(self, instance: EvalInstance) -> RemoteWorkspace:
        """Create and return a context-managed Workspace for the given instance."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """Run evaluation for a single instance in the provided workspace."""
        raise NotImplementedError

    def _create_error_output(
        self, instance: EvalInstance, error: Exception, retry_count: int
    ) -> EvalOutput:
        """Create an EvalOutput object for a failed instance."""
        return EvalOutput(
            instance_id=instance.id,
            test_result={},
            instruction=None,
            error=(
                f"Instance failed after {retry_count} retries. Last error: {str(error)}"
            )[:200],
            history=[],
            instance=instance.data,
        )

    # --- Runner ---
    def run(
        self,
        *,
        on_result: Optional[OnResult] = None,
    ) -> List[EvalOutput]:
        """
        Run evaluation with iterative mode support.

        If max_attempts > 1, will retry failed instances multiple times.
        If max_attempts == 1, will run once without retries.
        """
        logger.info("Starting evaluation (process pool)")
        logger.info("metadata=%s", self.metadata)
        logger.info("workers=%d", self.num_workers)
        logger.info("max_attempts=%d", self.metadata.max_attempts)

        # Use iterative mode for all cases
        return self._run_iterative_mode(on_result=on_result)

    def _get_resume_start_attempt(self) -> Tuple[int, List[EvalOutput]]:
        """
        Find where to resume and load previous outputs.

        Returns:
            Tuple of (start_attempt, previous_outputs)
            - start_attempt: Which attempt to start from (1 for fresh start)
            - previous_outputs: All outputs from previous attempts
        """
        all_previous_outputs = []

        # Check backwards from max_attempts to find the last attempt with results
        for attempt in range(self.metadata.max_attempts, 0, -1):
            attempt_file = os.path.join(
                self.metadata.eval_output_dir, f"output.critic_attempt_{attempt}.jsonl"
            )
            if os.path.exists(attempt_file) and os.path.getsize(attempt_file) > 0:
                # Found the last attempt with results, resume from here
                logger.info(f"Found existing results up to attempt {attempt}")

                # Load ALL previous outputs from attempts 1 to attempt
                for a in range(1, attempt + 1):
                    a_file = os.path.join(
                        self.metadata.eval_output_dir,
                        f"output.critic_attempt_{a}.jsonl",
                    )
                    if os.path.exists(a_file):
                        try:
                            with open(a_file, "r", encoding="utf-8") as f:
                                for line in f:
                                    if line.strip():
                                        output = EvalOutput.model_validate(
                                            json.loads(line)
                                        )
                                        all_previous_outputs.append(output)
                        except Exception as e:
                            logger.warning(f"Error loading outputs from {a_file}: {e}")

                logger.info(f"Loaded {len(all_previous_outputs)} previous outputs")
                return attempt, all_previous_outputs

        # No existing files found, start fresh
        logger.info("No existing results found, starting fresh")
        return 1, []

    def _run_iterative_mode(
        self,
        *,
        on_result: Optional[OnResult] = None,
    ) -> List[EvalOutput]:
        """Run evaluation with support for single or multiple attempts."""
        # Get all instances first
        all_instances = self.prepare_instances()

        total_instances = len(all_instances)
        logger.info("prepared %d instances for evaluation", total_instances)

        if total_instances == 0:
            logger.warning("No instances to process.")
            return []

        # Check for resume point and load previous outputs
        start_attempt, all_outputs = self._get_resume_start_attempt()

        for attempt in range(start_attempt, self.metadata.max_attempts + 1):
            logger.info(f"Starting attempt {attempt}/{self.metadata.max_attempts}")

            # Determine what this attempt should process
            if attempt == 1:
                target_instances = set(inst.id for inst in all_instances)
            else:
                prev_file = os.path.join(
                    self.metadata.eval_output_dir,
                    f"output.critic_attempt_{attempt - 1}.jsonl",
                )
                if os.path.exists(prev_file):
                    target_instances = get_failed_instances(
                        prev_file, self.metadata.critic
                    )
                else:
                    target_instances = set()

            # Exclude already completed in current attempt
            completed = get_completed_instances(
                os.path.join(
                    self.metadata.eval_output_dir,
                    f"output.critic_attempt_{attempt}.jsonl",
                )
            )
            instances_to_process = [
                inst
                for inst in all_instances
                if inst.id in target_instances and inst.id not in completed
            ]

            logger.info(f"Processing {len(instances_to_process)} instances")

            if not instances_to_process:
                logger.info("No instances to process, skipping to next attempt")
                continue

            # Adjust temperature for retries (deterministic -> non-deterministic)
            original_temperature = self.metadata.llm.temperature
            if attempt > 1 and original_temperature == 0.0:
                logger.info("Adjusting temperature from 0.0 to 0.1 for retry attempt")
                self.metadata.llm.temperature = 0.1

            # Create attempt-specific output callback
            attempt_outputs: List[EvalOutput] = []

            def attempt_on_result(instance: EvalInstance, out: EvalOutput) -> None:
                attempt_outputs.append(out)
                # Write to attempt-specific file
                attempt_file = os.path.join(
                    self.metadata.eval_output_dir,
                    f"output.critic_attempt_{attempt}.jsonl",
                )
                try:
                    with open(attempt_file, "a") as f:
                        f.write(out.model_dump_json() + "\n")
                except Exception as e:
                    logger.warning(
                        f"Failed to write to attempt file {attempt_file}: {e}"
                    )

                # Call original callback if provided
                if on_result:
                    try:
                        on_result(instance, out)
                    except Exception as cb_err:
                        logger.warning("on_result callback failed: %s", cb_err)

            # Run evaluation for this attempt
            pool = ProcessPoolExecutor(max_workers=self.num_workers)
            futures = []
            try:
                futures = [
                    pool.submit(self._process_one_mp, inst)
                    for inst in instances_to_process
                ]

                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Attempt {attempt}",
                    leave=False,
                ):
                    try:
                        instance, out = fut.result()
                        attempt_on_result(instance, out)
                    except Exception as e:
                        logger.error(
                            f"Unexpected error from worker process: {str(e)[:50]}",
                            exc_info=True,
                            stack_info=True,
                        )

                # Normal completion - shutdown gracefully
                pool.shutdown(wait=True)
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt received, shutting down workers...")
                self._cleanup_pool(pool, futures, wait=False)
                logger.info("All workers terminated")
                raise
            except Exception:
                self._cleanup_pool(pool, futures, wait=False)
                raise

            # Restore original temperature
            if attempt > 1 and original_temperature == 0.0:
                self.metadata.llm.temperature = original_temperature

            logger.info(
                f"Attempt {attempt} complete: "
                f"{len(attempt_outputs)} instances processed"
            )
            all_outputs.extend(attempt_outputs)

        # Aggregate results from all attempts
        logger.info("Aggregating results from all attempts")
        aggregate_results(
            output_dir=self.metadata.eval_output_dir,
            max_attempts=self.metadata.max_attempts,
            critic=self.metadata.critic,
            final_output_file="output.jsonl",
        )

        logger.info(
            f"Evaluation complete: {total_instances} total instances, "
            f"{self.metadata.max_attempts} max attempts"
        )
        return all_outputs

    def _cleanup_pool(
        self,
        pool: ProcessPoolExecutor,
        futures: list,
        wait: bool = False,
    ) -> None:
        """Clean up pool by canceling futures, terminating workers, and shutting down.

        Args:
            pool: The ProcessPoolExecutor to clean up
            futures: List of futures to cancel
            wait: Whether to wait for workers to finish (True) or terminate immediately (False)
        """
        # Cancel all pending futures
        for fut in futures:
            fut.cancel()

        # Forcefully terminate all worker processes if not waiting
        if not wait and hasattr(pool, "_processes") and pool._processes:
            for process in pool._processes.values():
                try:
                    process.terminate()
                except Exception:
                    pass

        # Shutdown the pool
        pool.shutdown(wait=wait, cancel_futures=True)

    # --- Worker-side method (executed in child processes) ---------------------------
    def _process_one_mp(
        self, instance: EvalInstance
    ) -> Tuple[EvalInstance, EvalOutput]:
        """Execute one instance in a child process with retry logic.

        - Creates workspace in the *child* process
        - Handles retries within the worker process
        - Ensures proper context-managed cleanup
        - Returns (instance, output) so the parent can stream results
        """
        # Set up instance-specific logging
        log_dir = os.path.join(self.metadata.eval_output_dir, "logs")
        reset_logger_for_multiprocessing(log_dir, instance.id)

        # Get log file path for stdout/stderr redirection
        log_file = os.path.join(log_dir, f"instance_{instance.id}.output.log")

        # Redirect stdout/stderr to capture all output (SDK visualizations, etc.)
        with redirect_stdout_stderr(log_file):
            logger.info("[child] start id=%s", instance.id)

            retry_count = 0
            last_error = None
            max_retries = self.metadata.max_retries

            while retry_count <= max_retries:
                workspace = None
                try:
                    workspace = self.prepare_workspace(instance)
                    out = self.evaluate_instance(instance, workspace)
                    logger.info("[child] done id=%s", instance.id)
                    return instance, out
                except Exception as e:
                    last_error = e
                    retry_count += 1

                    if retry_count <= max_retries:
                        logger.warning(
                            f"[child] Instance {instance.id} failed "
                            f"(attempt {retry_count}/{max_retries}): "
                            f"{str(e)[:50]}"
                        )
                    else:
                        logger.error(
                            f"[child] Instance {instance.id} failed after "
                            f"{max_retries} retries. Last error: {str(e)[:50]}",
                            exc_info=True,
                        )
                        # Create error output for final failure
                        error_output = self._create_error_output(
                            instance, last_error, max_retries
                        )
                        return instance, error_output
                finally:
                    # Ensure workspace cleanup happens regardless of success or failure
                    if workspace is not None:
                        try:
                            # Use the context manager protocol for cleanup
                            workspace.__exit__(None, None, None)
                            logger.debug(
                                "[child] cleaned up workspace for id=%s", instance.id
                            )
                        except Exception as cleanup_error:
                            logger.warning(
                                f"[child] Failed to cleanup workspace for {instance.id}: "
                                f"{str(cleanup_error)[:50]}"
                            )

            # This should never be reached, but added for type safety
            error_output = self._create_error_output(
                instance, Exception("Unexpected error: no attempts made"), max_retries
            )
            return instance, error_output


# ---------- Multiprocessing logging helper ---------------------------------------


def reset_logger_for_multiprocessing(log_dir: str, instance_id: str) -> None:
    """Reset the logger for multiprocessing with instance-specific logging.

    Save logs to a separate file for each instance, instead of trying to write to the
    same file/console from multiple processes. This provides:
    - One INFO line to console at start with tail hint
    - All subsequent logs go to instance-specific file
    - Only WARNING+ messages go to console after initial message

    Args:
        log_dir: Directory to store log files
        instance_id: Unique identifier for the instance being processed
    """
    import logging

    # Set up logger
    log_file = os.path.join(log_dir, f"instance_{instance_id}.log")

    # Get root logger and remove all existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler for initial message
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            f"Instance {instance_id} - " + "%(asctime)s - %(levelname)s - %(message)s"
        )
    )
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)

    # Print one INFO line with helpful hint
    root_logger.info(
        f"Starting evaluation for instance {instance_id}.\n"
        f'Hint: run "tail -f {log_file}" to see live logs in a separate shell'
    )

    # Now set console to WARNING+ only
    console_handler.setLevel(logging.WARNING)

    # Add file handler for detailed logs
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)


@contextmanager
def redirect_stdout_stderr(log_file_path: str):
    """Context manager to redirect stdout/stderr to a log file.

    This captures all print() statements, SDK visualizations, and any other
    output that goes to stdout/stderr.

    Args:
        log_file_path: Path to the log file where output should be redirected
    """
    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = None

    try:
        # Open log file in append mode with line buffering
        log_file = open(log_file_path, "a", buffering=1, encoding="utf-8")

        # Redirect stdout and stderr
        sys.stdout = log_file
        sys.stderr = log_file

        yield

    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Close the log file if it was opened
        if log_file is not None and not log_file.closed:
            log_file.close()
