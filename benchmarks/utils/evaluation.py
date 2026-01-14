"""
Evaluation orchestrator.
"""

import base64
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from uuid import UUID

import numpy as np
from lmnr import Laminar
from pydantic import BaseModel, Field
from tqdm import tqdm

from benchmarks.utils.constants import OUTPUT_FILENAME
from benchmarks.utils.conversation import CONVERSATION_EVENT_LOGGING_ENV_VAR
from benchmarks.utils.critics import get_completed_instances
from benchmarks.utils.iterative import aggregate_results, get_failed_instances
from benchmarks.utils.laminar import LMNR_ENV_VARS, LaminarEvalMetadata, LaminarService
from benchmarks.utils.models import (
    EvalInstance,
    EvalInstanceID,
    EvalMetadata,
    EvalOutput,
    RemoteRuntimeAllocation,
)
from openhands.sdk import get_logger
from openhands.sdk.critic import CriticBase
from openhands.sdk.workspace import RemoteWorkspace
from openhands.workspace import APIRemoteWorkspace


logger = get_logger(__name__)

OnResult = Callable[[EvalInstance, EvalOutput], None]


class Evaluation(ABC, BaseModel):
    """Abstract orchestrator for instance processing (process-based)."""

    metadata: EvalMetadata
    num_workers: int = Field(default=1, ge=1)
    current_attempt: int = Field(
        default=1, description="Current attempt number (1-indexed)"
    )

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
    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """Create and return a context-managed Workspace for the given instance.

        Args:
            instance: The evaluation instance to prepare workspace for.
            resource_factor: Resource factor for runtime allocation (default: 1).
            forward_env: Environment variables to forward into the workspace.
        """
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

        # Initialize Laminar
        LaminarService.get().initialize()

        # Create Laminar evaluation
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.metadata.lmnr = LaminarEvalMetadata(
            eval_id=LaminarService.get().create_evaluation(
                name=f"{self.metadata.dataset} {self.metadata.dataset_split} {now}",
                group_name=f"{self.metadata.dataset} {self.metadata.dataset_split}",
                metadata=self.metadata.model_dump(mode="json"),
            )
        )

        total_instances = len(all_instances)
        logger.info("prepared %d instances for evaluation", total_instances)

        if total_instances == 0:
            logger.warning("No instances to process.")
            return []

        critic = self.metadata.critic
        all_outputs: List[EvalOutput] = []

        for attempt in range(1, self.metadata.max_attempts + 1):
            self.current_attempt = attempt
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

            def _make_json_safe(value: object) -> object:
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, dict):
                    return {k: _make_json_safe(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [_make_json_safe(v) for v in value]
                return value

            def attempt_on_result(instance: EvalInstance, out: EvalOutput) -> None:
                attempt_outputs.append(out)
                # Write to attempt-specific file
                attempt_file = os.path.join(
                    self.metadata.eval_output_dir,
                    f"output.critic_attempt_{attempt}.jsonl",
                )
                try:
                    payload = _make_json_safe(out.model_dump(mode="json"))
                    with open(attempt_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(payload) + "\n")
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
                futures = []
                lmnr_datapoints: dict[str, UUID] = dict()
                for index, inst in enumerate(instances_to_process):
                    datapoint_id, lmnr_span_ctx = (
                        LaminarService.get().create_evaluation_datapoint(
                            self.metadata.lmnr.eval_id,
                            inst.id,
                            self.metadata.model_dump(mode="json"),
                            index,
                        )
                    )
                    if datapoint_id is not None:
                        lmnr_datapoints[inst.id] = datapoint_id

                    futures.append(
                        pool.submit(self._process_one_mp, inst, lmnr_span_ctx, attempt)
                    )

                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Attempt {attempt}",
                    leave=False,
                ):
                    try:
                        instance, out = fut.result()

                        # Add Laminar metadata to EvalOutput so we can use it in the evaluation process
                        if out.metadata is None:
                            out.metadata = self.metadata.model_copy(deep=True)
                        out.metadata.lmnr = LaminarEvalMetadata(
                            eval_id=self.metadata.lmnr.eval_id,
                            datapoint_id=lmnr_datapoints.get(instance.id, None),
                        )

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

    def _calculate_resource_factor(self, runtime_failure_count: int) -> int:
        """Calculate the resource factor based on runtime failure count.

        Uses exponential backoff: base_factor * 2^runtime_failure_count
        Capped at max_resource_factor from metadata.

        Args:
            runtime_failure_count: Number of runtime failures encountered so far.

        Returns:
            The resource factor to use for this attempt.
        """
        if runtime_failure_count <= 0:
            return self.metadata.base_resource_factor

        factor = self.metadata.base_resource_factor * (2**runtime_failure_count)
        return min(factor, self.metadata.max_resource_factor)

    # --- Worker-side method (executed in child processes) ---------------------------
    def _process_one_mp(
        self, instance: EvalInstance, eval_span_ctx: str | None, critic_attempt: int
    ) -> Tuple[EvalInstance, EvalOutput]:
        """Execute one instance in a child process with retry logic.

        - Creates workspace in the *child* process
        - Handles retries within the worker process
        - Tracks runtime failures and increases resource_factor exponentially
        - Ensures proper context-managed cleanup
        - Returns (instance, output) so the parent can stream results
        """
        timeline_dir = Path(self.metadata.eval_output_dir) / "timelines"
        timeline_dir.mkdir(parents=True, exist_ok=True)

        def write_timeline(entry: dict[str, object]) -> None:
            path = timeline_dir / f"{instance.id}.attempt{entry.get('attempt', 0)}.json"
            path.write_text(json.dumps(entry, indent=2))

        # Set up instance-specific logging
        log_dir = os.path.join(self.metadata.eval_output_dir, "logs")
        reset_logger_for_multiprocessing(log_dir, instance.id)

        # Get log file path for stdout/stderr redirection
        log_file = os.path.join(log_dir, f"instance_{instance.id}.output.log")

        # Redirect stdout/stderr to capture all output (SDK visualizations, etc.)
        with redirect_stdout_stderr(log_file):
            logger.info("[child] start id=%s", instance.id)

            retry_count = 0
            runtime_failure_count = 0
            last_error = None
            max_retries = self.metadata.max_retries
            runtime_runs: list[RemoteRuntimeAllocation] = []

            while retry_count <= max_retries:
                workspace = None
                attempt_index = retry_count + 1
                attempt_start_ns = time.perf_counter_ns()
                attempt_start_ts = datetime.now(timezone.utc).isoformat()
                attempt_status = "error"
                phases: list[dict[str, int | str]] = []
                resource_factor = self.metadata.base_resource_factor

                # Start Laminar execution span and inject context into os.environ so workspace can pick it up
                # Escape the serialized context to safely pass as a cli argument
                lmnr_span = Laminar.start_active_span(
                    "Execution",
                    span_type="EXECUTOR",  # type: ignore
                    parent_span_context=Laminar.deserialize_span_context(eval_span_ctx)
                    if eval_span_ctx
                    else None,
                )
                exec_span_ctx = json.dumps(Laminar.serialize_span_context(lmnr_span))
                os.environ["LMNR_SPAN_CONTEXT"] = exec_span_ctx or ""

                try:
                    # Calculate resource factor based on runtime failures
                    resource_factor = self._calculate_resource_factor(
                        runtime_failure_count
                    )
                    if runtime_failure_count > 0:
                        logger.warning(
                            f"[child] Instance {instance.id}: "
                            f"attempt {retry_count + 1}/{max_retries + 1}, "
                            f"runtime_failure_count={runtime_failure_count}, "
                            f"resource_factor={resource_factor}"
                        )

                    ws_start = time.perf_counter_ns()
                    workspace = self.prepare_workspace(
                        instance,
                        resource_factor=resource_factor,
                        forward_env=LMNR_ENV_VARS,
                    )
                    phases.append(
                        {
                            "name": "prepare_workspace",
                            "start_ns": int(ws_start),
                            "end_ns": int(time.perf_counter_ns()),
                        }
                    )

                    # Record runtime/pod mapping only for remote runtimes
                    if isinstance(workspace, APIRemoteWorkspace):
                        retry_number = retry_count + 1  # 1-indexed for readability
                        runtime_run = RemoteRuntimeAllocation(
                            runtime_id=getattr(workspace, "_runtime_id", None),
                            session_id=getattr(workspace, "session_id", None),
                            runtime_url=getattr(workspace, "_runtime_url", None),
                            resource_factor=resource_factor,
                            critic_attempt=critic_attempt,
                            retry=retry_number,
                            started_at=datetime.now(timezone.utc),
                        )
                        runtime_runs.append(runtime_run)
                        logger.info(
                            "[child] runtime allocated instance=%s attempt=%d retry=%d workspace=%s runtime_id=%s session_id=%s resource_factor=%s",
                            instance.id,
                            critic_attempt,
                            retry_number,
                            workspace.__class__.__name__,
                            runtime_run.runtime_id,
                            runtime_run.session_id,
                            runtime_run.resource_factor,
                        )
                    eval_start = time.perf_counter_ns()
                    out = self.evaluate_instance(instance, workspace)
                    phases.append(
                        {
                            "name": "evaluate_instance",
                            "start_ns": int(eval_start),
                            "end_ns": int(time.perf_counter_ns()),
                        }
                    )
                    if runtime_runs:
                        out.runtime_runs = runtime_runs
                    logger.info("[child] done id=%s", instance.id)
                    attempt_status = "ok"
                    return instance, out
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    lmnr_span.record_exception(e)

                    # Log structured runtime allocation/init failures so we can trace instance -> runtime/pod
                    runtime_id = (
                        getattr(workspace, "_runtime_id", None) if workspace else None
                    )
                    session_id = (
                        getattr(workspace, "session_id", None) if workspace else None
                    )
                    if isinstance(workspace, APIRemoteWorkspace) or (
                        "Runtime not yet ready" in str(e)
                    ):
                        logger.warning(
                            "[child] runtime init failure instance=%s attempt=%d retry=%d runtime_id=%s session_id=%s error=%s",
                            instance.id,
                            critic_attempt,
                            retry_count,
                            runtime_id,
                            session_id,
                            str(e),
                        )

                    # TODO(#277): add an exception classifier to decide when to bump resources
                    runtime_failure_count += 1
                    logger.warning(
                        f"[child] Instance {instance.id}: runtime_failure_count="
                        f"{runtime_failure_count}"
                    )

                    if retry_count <= max_retries:
                        logger.warning(
                            f"[child] Instance {instance.id} failed "
                            f"(attempt {retry_count}/{max_retries}): "
                            f"{str(e)}"
                        )
                    else:
                        logger.error(
                            f"[child] Instance {instance.id} failed after "
                            f"{max_retries} retries. Last error: {str(e)}",
                            exc_info=True,
                        )
                        # Create error output for final failure
                        error_output = self._create_error_output(
                            instance, last_error, max_retries
                        )
                        if runtime_runs:
                            error_output.runtime_runs = runtime_runs
                        return instance, error_output
                finally:
                    # Ensure workspace cleanup happens regardless of success or failure
                    cleanup_start = time.perf_counter_ns()
                    if workspace is not None:
                        try:
                            self._capture_conversation_archive(workspace, instance)
                        except Exception as archive_error:
                            logger.warning(
                                "[child] Failed to capture conversation archive for %s: %s",
                                instance.id,
                                archive_error,
                            )
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
                    lmnr_span.end()
                    phases.append(
                        {
                            "name": "cleanup",
                            "start_ns": int(cleanup_start),
                            "end_ns": int(time.perf_counter_ns()),
                        }
                    )
                    attempt_end_ns = time.perf_counter_ns()
                    write_timeline(
                        {
                            "instance_id": instance.id,
                            "attempt": attempt_index,
                            "critic_attempt": critic_attempt,
                            "status": attempt_status,
                            "error": (
                                str(last_error) if attempt_status != "ok" else None
                            ),
                            "start_ts": attempt_start_ts,
                            "end_ts": datetime.now(timezone.utc).isoformat(),
                            "duration_ms": (attempt_end_ns - attempt_start_ns)
                            / 1_000_000,
                            "resource_factor": resource_factor,
                            "runtime_failure_count": runtime_failure_count,
                            "phases": [
                                {
                                    "name": p["name"],
                                    "duration_ms": (
                                        (int(p["end_ns"]) - int(p["start_ns"]))
                                        / 1_000_000
                                    ),
                                    "start_ns": int(p["start_ns"]),
                                    "end_ns": int(p["end_ns"]),
                                }
                                for p in phases
                            ],
                        }
                    )

            # This should never be reached, but added for type safety
            error_output = self._create_error_output(
                instance, Exception("Unexpected error: no attempts made"), max_retries
            )
            if runtime_runs:
                error_output.runtime_runs = runtime_runs
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

    class ConversationEventFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            msg = record.getMessage()
            return msg in {"conversation_event", "conversation_event_metadata"}

    # Datadog/console handler for conversation events (bypasses stdout redirection)
    if bool(os.environ.get(CONVERSATION_EVENT_LOGGING_ENV_VAR, False)):
        from pythonjsonlogger.json import JsonFormatter

        dd_handler = logging.StreamHandler(sys.__stdout__)
        dd_handler.setLevel(logging.INFO)
        dd_handler.addFilter(ConversationEventFilter())
        dd_handler.setFormatter(
            JsonFormatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(run_id)s %(instance_id)s %(attempt)s %(event_type)s %(event_size)s"
            )
        )
        root_logger.addHandler(dd_handler)

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
