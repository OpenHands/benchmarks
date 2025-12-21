"""
Evaluation orchestrator.
"""

import base64
import json
import os
import shutil
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel, Field
from tqdm import tqdm

from benchmarks.utils.constants import OUTPUT_FILENAME
from benchmarks.utils.critics import evaluate_output, get_completed_instances
from benchmarks.utils.iterative import get_failed_instances
from benchmarks.utils.models import (
    EvalInstance,
    EvalInstanceID,
    EvalMetadata,
    EvalOutput,
    cost_from_metrics,
    write_derived_report,
    write_output_line,
)
from openhands.sdk import get_logger
from openhands.sdk.critic import CriticBase
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
        self, instance: EvalInstance, workspace: RemoteWorkspace, attempt: int
    ) -> EvalOutput:
        """Run evaluation for a single instance in the provided workspace."""
        raise NotImplementedError

    def _create_error_output(
        self,
        instance: EvalInstance,
        error: Exception,
        retry_count: int,
        attempt: int,
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
            status="error",
            resolved=None,
            attempt=attempt,
            max_attempts=self.metadata.max_attempts,
            cost=cost_from_metrics(None),
            artifacts_url="",
        )

    def _capture_conversation_archive(
        self,
        workspace: RemoteWorkspace,
        instance: EvalInstance,
    ) -> None:
        """Capture conversation trajectory from the remote runtime.

        Persists the /workspace/conversations directory from the remote runtime
        to a per-instance tar.gz file in the evaluation output directory.

        This provides a complete record of the agent's conversation history,
        which is valuable for debugging, analysis, and reproducibility.

        Args:
            workspace: The remote workspace to capture from
            instance: The evaluation instance being processed
        """
        try:
            # Create command to tar and base64 encode the conversations directory
            conv_cmd = (
                "cd / && "
                "if [ -d workspace/conversations ]; then "
                "tar -czf - workspace/conversations | base64; "
                "else echo ''; fi"
            )
            tar_cmd = workspace.execute_command(conv_cmd)

            if tar_cmd.exit_code == 0 and tar_cmd.stdout.strip():
                # Save to instance-specific file to support parallel execution
                conversations_dir = (
                    Path(self.metadata.eval_output_dir) / "conversations"
                )
                conversations_dir.mkdir(parents=True, exist_ok=True)
                conv_tar_path = conversations_dir / f"{instance.id}.tar.gz"

                # Decode and write the tar.gz file
                conv_tar_path.write_bytes(base64.b64decode(tar_cmd.stdout))
                logger.info(
                    "[child] Saved conversation archive for %s to %s",
                    instance.id,
                    conv_tar_path,
                )
            else:
                logger.debug(
                    "[child] No conversation archive for %s (directory not found or empty)",
                    instance.id,
                )
        except Exception as e:
            logger.warning(
                "[child] Failed to capture conversation trajectory for %s: %s",
                instance.id,
                e,
            )

    def _stage_instance_artifacts(
        self, instance_id: str, attempt: int, out: EvalOutput
    ) -> str:
        """Bundle per-instance artifacts under artifacts/ for canonical output."""
        base_dir = Path(self.metadata.eval_output_dir)
        attempt_dir = base_dir / "artifacts" / instance_id / f"attempt_{attempt}"
        try:
            attempt_dir.mkdir(parents=True, exist_ok=True)

            logs_dir = attempt_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            log_source = base_dir / "logs" / f"instance_{instance_id}.log"
            output_log_source = base_dir / "logs" / f"instance_{instance_id}.output.log"
            if log_source.exists():
                shutil.copy2(log_source, logs_dir / "instance.log")
            if output_log_source.exists():
                shutil.copy2(output_log_source, logs_dir / "instance.output.log")

            conv_source = base_dir / "conversations" / f"{instance_id}.tar.gz"
            if conv_source.exists():
                shutil.copy2(conv_source, attempt_dir / "conversation.tar.gz")

            if out.test_result:
                with open(attempt_dir / "test_result.json", "w", encoding="utf-8") as f:
                    json.dump(out.test_result, f, indent=2)

            if out.artifacts:
                with open(attempt_dir / "artifacts.json", "w", encoding="utf-8") as f:
                    json.dump(out.artifacts, f, indent=2)

            if out.error:
                (attempt_dir / "error.txt").write_text(out.error, encoding="utf-8")
        except Exception as e:
            logger.warning(
                "Failed to stage artifacts for %s attempt %s: %s",
                instance_id,
                attempt,
                e,
            )
            return ""

        return str(attempt_dir.relative_to(base_dir))

    def _prepare_output_for_jsonl(
        self, instance: EvalInstance, out: EvalOutput, attempt: int
    ) -> None:
        """Populate canonical fields before writing output.jsonl."""
        status = "error" if out.error else (out.status or "success")
        if out.resolved is not None:
            resolved = out.resolved
        elif status == "error":
            resolved = None
        else:
            resolved = evaluate_output(self.metadata.critic, out)

        artifacts_url = self._stage_instance_artifacts(instance.id, attempt, out)

        out.attempt = attempt
        out.max_attempts = self.metadata.max_attempts
        out.status = status
        out.resolved = resolved
        out.cost = cost_from_metrics(out.metrics)
        out.artifacts_url = artifacts_url

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

    def _get_instances_for_attempt(
        self,
        attempt: int,
        all_instances: List[EvalInstance],
        critic: CriticBase,
    ) -> List[EvalInstance]:
        """
        Determine which instances need processing for a specific attempt.

        This method handles all resume scenarios naturally without special cases:
        - New instances: Not completed in attempt 1 yet → include them
        - Resume: Already completed in this attempt → exclude them
        - Expansion: Just more instances not in attempt 1 yet → include them

        Args:
            attempt: The attempt number (1-indexed)
            all_instances: All instances in the dataset
            critic: The critic to use for determining failures

        Returns:
            List of instances that need processing for this attempt
        """
        attempt_file = os.path.join(
            self.metadata.eval_output_dir,
            f"output.critic_attempt_{attempt}.jsonl",
        )
        completed_in_attempt = get_completed_instances(attempt_file)

        if attempt == 1:
            # Attempt 1: Process everything not yet completed in attempt 1
            return [
                inst for inst in all_instances if inst.id not in completed_in_attempt
            ]
        else:
            # Attempt N: Process what failed in N-1 and isn't completed in N
            prev_file = os.path.join(
                self.metadata.eval_output_dir,
                f"output.critic_attempt_{attempt - 1}.jsonl",
            )
            if not os.path.exists(prev_file):
                return []

            failed_in_prev = get_failed_instances(prev_file, critic)
            return [
                inst
                for inst in all_instances
                if inst.id in failed_in_prev and inst.id not in completed_in_attempt
            ]

    def _run_iterative_mode(
        self,
        *,
        on_result: Optional[OnResult] = None,
    ) -> List[EvalOutput]:
        """Run evaluation with support for single or multiple attempts."""
        all_instances = self.prepare_instances()

        total_instances = len(all_instances)
        logger.info("prepared %d instances for evaluation", total_instances)

        if total_instances == 0:
            logger.warning("No instances to process.")
            return []

        critic = self.metadata.critic
        all_outputs: List[EvalOutput] = []

        for attempt in range(1, self.metadata.max_attempts + 1):
            logger.info(f"Starting attempt {attempt}/{self.metadata.max_attempts}")

            instances_to_process = self._get_instances_for_attempt(
                attempt, all_instances, critic
            )

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
                attempt_file = Path(
                    self.metadata.eval_output_dir,
                    f"output.critic_attempt_{attempt}.jsonl",
                )
                try:
                    self._prepare_output_for_jsonl(instance, out, attempt)
                    with open(attempt_file, "a", encoding="utf-8") as f:
                        f.write(out.model_dump_json() + "\n")
                    write_output_line(Path(self.output_path), out)
                except Exception as e:
                    logger.warning("Failed to write output for %s: %s", instance.id, e)

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
                    pool.submit(self._process_one_mp, inst, attempt)
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

        logger.info("Writing derived report from output.jsonl")
        try:
            write_derived_report(self.metadata.eval_output_dir)
        except Exception as e:
            logger.warning("Failed to write derived report: %s", e)

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
        self, instance: EvalInstance, attempt: int
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
                    out = self.evaluate_instance(instance, workspace, attempt)

                    # Capture conversation archive after successful evaluation
                    self._capture_conversation_archive(workspace, instance)

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
                            instance, last_error, max_retries, attempt
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
                instance,
                Exception("Unexpected error: no attempts made"),
                max_retries,
                attempt,
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
    output_log_file = os.path.join(log_dir, f"instance_{instance_id}.output.log")

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
        f"""
    === Evaluation Started (instance {instance_id}) ===
    View live output:
    • tail -f {log_file}          (logger)
    • tail -f {output_log_file}   (stdout/stderr)
    ===============================================
    """.strip()
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
