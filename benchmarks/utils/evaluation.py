"""
Evaluation orchestrator.

This module provides async-based evaluation orchestration for benchmarks.
The evaluation uses asyncio for concurrent instance processing, running
synchronous SDK operations in thread executors. This eliminates the 30×
memory multiplication from ProcessPoolExecutor while maintaining high
concurrency for I/O-bound workloads (HTTP calls to LLM proxy + runtime API).
"""

import asyncio
import base64
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, List, Optional, Tuple
from uuid import UUID

from lmnr import Laminar
from pydantic import BaseModel, Field
from tqdm import tqdm

from benchmarks.utils.constants import OUTPUT_FILENAME
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

# Interval in seconds between checking for per-instance timeouts
TIMEOUT_CHECK_INTERVAL_SECONDS = 60


@dataclass
class PendingInstance:
    """Tracks state for a pending evaluation instance."""

    instance: EvalInstance
    start_time: float
    datapoint_id: UUID | None = None
    task: asyncio.Task | None = field(default=None, repr=False)


OnResult = Callable[[EvalInstance, EvalOutput], None]


class Evaluation(ABC, BaseModel):
    """Abstract orchestrator for instance processing using asyncio.

    Uses asyncio for concurrent instance processing with a semaphore to limit
    the number of concurrent instances. Synchronous SDK operations (workspace,
    conversation) are run in thread executors via asyncio.to_thread().

    This design eliminates the memory multiplication from ProcessPoolExecutor
    while maintaining high concurrency for I/O-bound workloads.
    """

    metadata: EvalMetadata
    num_workers: int = Field(default=1, ge=1)
    current_attempt: int = Field(
        default=1, description="Current attempt number (1-indexed)"
    )
    instance_timeout: int = Field(
        default=4 * 60 * 60,  # 4 hours
        description=(
            "Maximum time in seconds for a single instance to complete. "
            "When a timeout occurs, the instance's asyncio task is cancelled. "
            "The underlying thread running the SDK operation will complete, "
            "but the result will be discarded and replaced with a timeout error."
        ),
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
                    "[worker] Saved conversation archive for %s to %s",
                    instance.id,
                    conv_tar_path,
                )
            else:
                logger.debug(
                    "[worker] No conversation archive for %s (directory not found or empty)",
                    instance.id,
                )
        except Exception as e:
            logger.warning(
                "[worker] Failed to capture conversation trajectory for %s: %s",
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

        Uses asyncio for concurrent instance processing. Synchronous SDK
        operations run in thread executors via asyncio.to_thread().
        """
        logger.info("Starting evaluation (asyncio)")
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
        """Run evaluation with support for single or multiple attempts.

        Uses asyncio for concurrent instance processing. Synchronous SDK
        operations run in thread executors via asyncio.to_thread().
        """
        return asyncio.run(self._run_iterative_mode_async(on_result=on_result))

    async def _run_iterative_mode_async(
        self,
        *,
        on_result: Optional[OnResult] = None,
    ) -> List[EvalOutput]:
        """Async implementation of iterative mode evaluation."""
        all_instances = self.prepare_instances()

        # Initialize Laminar
        LaminarService.get().initialize()

        # Build metadata for Laminar evaluation and traces
        run_id = os.getenv("UNIQUE_EVAL_NAME")
        laminar_meta = {
            k: v
            for k, v in [
                ("benchmark", self.metadata.dataset),
                ("model", self.metadata.llm.model),
            ]
            if v
        }

        # Create Laminar evaluation (use run_id as name if available)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        eval_name = (
            run_id or f"{self.metadata.dataset} {self.metadata.dataset_split} {now}"
        )
        self.metadata.lmnr = LaminarEvalMetadata(
            eval_id=LaminarService.get().create_evaluation(
                name=eval_name,
                group_name=f"{self.metadata.dataset} {self.metadata.dataset_split}",
                metadata=laminar_meta or None,
            )
        )
        # Store for use in datapoint creation
        self._laminar_session_id = run_id
        self._laminar_trace_meta = laminar_meta or None

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

            # Create attempt-specific output callback and file write lock
            attempt_outputs: List[EvalOutput] = []
            file_lock = asyncio.Lock()

            async def attempt_on_result_async(
                instance: EvalInstance, out: EvalOutput
            ) -> None:
                # Write to attempt-specific file (thread-safe with lock)
                attempt_file = os.path.join(
                    self.metadata.eval_output_dir,
                    f"output.critic_attempt_{attempt}.jsonl",
                )
                async with file_lock:
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

                # Release heavy history data from memory now that it's
                # persisted to disk. The critic and aggregator read history
                # from the attempt files, not from this in-memory list.
                out.history = []

                attempt_outputs.append(out)

            # Run evaluation for this attempt using asyncio
            attempt_outputs = await self._run_attempt_async(
                instances_to_process,
                attempt,
                attempt_on_result_async,
            )

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

    async def _run_attempt_async(
        self,
        instances: List[EvalInstance],
        attempt: int,
        on_result: Callable[[EvalInstance, EvalOutput], Coroutine[Any, Any, None]],
    ) -> List[EvalOutput]:
        """Run a single attempt with async concurrency.

        Uses asyncio.Semaphore to limit concurrent instances and
        asyncio.to_thread() to run sync SDK operations.

        Args:
            instances: List of instances to process
            attempt: Current attempt number
            on_result: Async callback for each completed instance

        Returns:
            List of EvalOutput for completed instances
        """
        semaphore = asyncio.Semaphore(self.num_workers)
        pending_instances: dict[asyncio.Task, PendingInstance] = {}
        attempt_outputs: List[EvalOutput] = []
        progress = tqdm(total=len(instances), desc=f"Attempt {attempt}", leave=False)

        async def process_with_semaphore(
            inst: EvalInstance,
            lmnr_span_ctx: str | None,
            datapoint_id: UUID | None,
        ) -> Tuple[EvalInstance, EvalOutput]:
            """Process one instance with semaphore-based concurrency control."""
            async with semaphore:
                # Run the sync processing function in a thread
                return await asyncio.to_thread(
                    self._process_one_sync, inst, lmnr_span_ctx, attempt
                )

        # Create all tasks
        tasks: list[asyncio.Task] = []
        # lmnr is guaranteed to be set in _run_iterative_mode_async before this call
        assert self.metadata.lmnr is not None
        for index, inst in enumerate(instances):
            datapoint_id, lmnr_span_ctx = (
                LaminarService.get().create_evaluation_datapoint(
                    self.metadata.lmnr.eval_id,
                    inst.id,
                    self.metadata.model_dump(mode="json"),
                    index,
                    session_id=self._laminar_session_id,
                    trace_metadata=self._laminar_trace_meta,
                )
            )

            task = asyncio.create_task(
                process_with_semaphore(inst, lmnr_span_ctx, datapoint_id)
            )
            tasks.append(task)
            pending_instances[task] = PendingInstance(
                instance=inst,
                start_time=time.monotonic(),
                datapoint_id=datapoint_id,
                task=task,
            )

        # Process tasks as they complete with timeout checking
        pending: set[asyncio.Task] = set(tasks)

        while pending:
            # Wait for either a task to complete or timeout interval
            done, pending = await asyncio.wait(
                pending,
                timeout=TIMEOUT_CHECK_INTERVAL_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Process completed tasks
            for task in done:
                progress.update(1)
                pending_info = pending_instances.get(task)
                try:
                    instance, out = task.result()

                    # Add Laminar metadata to EvalOutput
                    if out.metadata is None:
                        out.metadata = self.metadata.model_copy(deep=True)
                    out.metadata.lmnr = LaminarEvalMetadata(
                        eval_id=self.metadata.lmnr.eval_id,
                        datapoint_id=(
                            pending_info.datapoint_id if pending_info else None
                        ),
                    )

                    await on_result(instance, out)
                    attempt_outputs.append(out)
                except asyncio.CancelledError:
                    # Task was cancelled due to timeout, error already handled
                    pass
                except Exception as e:
                    logger.error(
                        f"Unexpected error from task: {str(e)[:50]}",
                        exc_info=True,
                        stack_info=True,
                    )

            # Check for per-instance timeouts
            now = time.monotonic()
            timed_out_tasks = [
                task
                for task in pending
                if now - pending_instances[task].start_time > self.instance_timeout
            ]

            for task in timed_out_tasks:
                pending.discard(task)
                progress.update(1)
                pending_info = pending_instances.get(task)
                if pending_info:
                    inst = pending_info.instance
                    logger.error(
                        f"Instance {inst.id} timed out after {self.instance_timeout}s"
                    )
                    error_output = self._create_error_output(
                        inst,
                        TimeoutError(
                            f"Instance did not complete within "
                            f"{self.instance_timeout}s timeout"
                        ),
                        attempt,
                    )
                    if error_output.metadata is None:
                        error_output.metadata = self.metadata.model_copy(deep=True)
                    if self.metadata.lmnr is not None:
                        error_output.metadata.lmnr = LaminarEvalMetadata(
                            eval_id=self.metadata.lmnr.eval_id,
                            datapoint_id=pending_info.datapoint_id,
                        )
                    await on_result(inst, error_output)
                    attempt_outputs.append(error_output)
                # Cancel the task (the thread will continue but result is discarded)
                task.cancel()

        progress.close()
        return attempt_outputs

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

    # --- Worker method (executed in thread executor) ---------------------------
    def _process_one_sync(
        self, instance: EvalInstance, eval_span_ctx: str | None, critic_attempt: int
    ) -> Tuple[EvalInstance, EvalOutput]:
        """Execute one instance synchronously in a thread with retry logic.

        This method runs in a thread executor via asyncio.to_thread().
        It performs all sync SDK operations (workspace creation, conversation).

        - Creates workspace in the thread
        - Handles retries within the thread
        - Tracks runtime failures and increases resource_factor exponentially
        - Ensures proper context-managed cleanup
        - Returns (instance, output) so the async caller can stream results
        """
        # Set up instance-specific logging
        log_dir = os.path.join(self.metadata.eval_output_dir, "logs")
        setup_instance_logging(log_dir, instance.id)

        # Get log file path for stdout/stderr redirection
        log_file = os.path.join(log_dir, f"instance_{instance.id}.output.log")

        # Redirect stdout/stderr to capture all output (SDK visualizations, etc.)
        with redirect_stdout_stderr(log_file):
            logger.info("[worker] start id=%s", instance.id)

            retry_count = 0
            runtime_failure_count = 0
            last_error = None
            max_retries = self.metadata.max_retries
            runtime_runs: list[RemoteRuntimeAllocation] = []

            while retry_count <= max_retries:
                workspace = None

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
                            f"[worker] Instance {instance.id}: "
                            f"attempt {retry_count + 1}/{max_retries + 1}, "
                            f"runtime_failure_count={runtime_failure_count}, "
                            f"resource_factor={resource_factor}"
                        )

                    workspace = self.prepare_workspace(
                        instance,
                        resource_factor=resource_factor,
                        forward_env=LMNR_ENV_VARS,
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
                            "[worker] runtime allocated instance=%s attempt=%d retry=%d workspace=%s runtime_id=%s session_id=%s resource_factor=%s",
                            instance.id,
                            critic_attempt,
                            retry_number,
                            workspace.__class__.__name__,
                            runtime_run.runtime_id,
                            runtime_run.session_id,
                            runtime_run.resource_factor,
                        )
                    out = self.evaluate_instance(instance, workspace)
                    if runtime_runs:
                        out.runtime_runs = runtime_runs
                    logger.info("[worker] done id=%s", instance.id)
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
                            "[worker] runtime init failure instance=%s attempt=%d retry=%d runtime_id=%s session_id=%s error=%s",
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
                        f"[worker] Instance {instance.id}: runtime_failure_count="
                        f"{runtime_failure_count}"
                    )

                    if retry_count <= max_retries:
                        logger.warning(
                            f"[worker] Instance {instance.id} failed "
                            f"(attempt {retry_count}/{max_retries}): "
                            f"{str(e)}"
                        )
                    else:
                        logger.error(
                            f"[worker] Instance {instance.id} failed after "
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
                    if workspace is not None:
                        try:
                            self._capture_conversation_archive(workspace, instance)
                        except Exception as archive_error:
                            logger.warning(
                                "[worker] Failed to capture conversation archive for %s: %s",
                                instance.id,
                                archive_error,
                            )
                        try:
                            # Use the context manager protocol for cleanup
                            workspace.__exit__(None, None, None)
                            logger.debug(
                                "[worker] cleaned up workspace for id=%s", instance.id
                            )
                        except Exception as cleanup_error:
                            logger.warning(
                                f"[worker] Failed to cleanup workspace for {instance.id}: "
                                f"{str(cleanup_error)[:50]}"
                            )
                    lmnr_span.end()

            # This should never be reached, but added for type safety
            error_output = self._create_error_output(
                instance, Exception("Unexpected error: no attempts made"), max_retries
            )
            if runtime_runs:
                error_output.runtime_runs = runtime_runs
            return instance, error_output


# ---------- Logging helper for thread workers ------------------------------------


def setup_instance_logging(log_dir: str, instance_id: str) -> None:
    """Set up instance-specific logging for worker threads.

    See benchmarks.utils.console_logging.setup_instance_logging for details.
    """
    from benchmarks.utils.console_logging import (
        setup_instance_logging as _setup_logging,
    )

    _setup_logging(log_dir, instance_id)


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
