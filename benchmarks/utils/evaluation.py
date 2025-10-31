"""
Evaluation orchestrator.
"""

import json
import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel, Field
from tqdm import tqdm

from benchmarks.utils.constants import OUTPUT_FILENAME
from benchmarks.utils.critics import CriticRegistry, get_completed_instances
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
            history=None,
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
                                        output = EvalOutput(**json.loads(line))
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
        # Get all instances for the first attempt
        all_instances = self.prepare_instances()
        total_instances = len(all_instances)
        logger.info("prepared %d instances for evaluation", total_instances)

        if total_instances == 0:
            logger.warning("No instances to process.")
            return []

        # For single attempts without a critic, use the pass critic
        critic_name = self.metadata.critic_name
        if not critic_name:
            if self.metadata.max_attempts == 1:
                critic_name = "pass"
                logger.info(
                    "No critic specified for single attempt, using 'pass' critic"
                )
            else:
                raise ValueError("critic_name is required for multi-attempt evaluation")

        critic = CriticRegistry.create_critic(critic_name)

        # Check for resume point and load previous outputs
        start_attempt, all_outputs = self._get_resume_start_attempt()

        # Reconstruct instances_to_process for the resume attempt
        # Uniform logic for all attempts
        prev_attempt_file = os.path.join(
            self.metadata.eval_output_dir,
            f"output.critic_attempt_{start_attempt - 1}.jsonl",
        )

        if start_attempt == 1 or not os.path.exists(prev_attempt_file):
            # First attempt or no previous attempt exists: start with all instances
            target_instances = set(inst.id for inst in all_instances)
        else:
            # Start with failed instances from previous attempt
            target_instances = get_failed_instances(prev_attempt_file, critic)

        # For any attempt: exclude instances already completed in current attempt
        completed_in_current = get_completed_instances(
            os.path.join(
                self.metadata.eval_output_dir,
                f"output.critic_attempt_{start_attempt}.jsonl",
            )
        )

        instances_to_process = [
            inst
            for inst in all_instances
            if inst.id in target_instances and inst.id not in completed_in_current
        ]

        for attempt in range(start_attempt, self.metadata.max_attempts + 1):
            logger.info(f"Starting attempt {attempt}/{self.metadata.max_attempts}")
            logger.info(f"Processing {len(instances_to_process)} instances")

            if not instances_to_process:
                logger.info("No instances to process, stopping early")
                break

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
            with ProcessPoolExecutor(
                max_workers=self.num_workers, initializer=_child_init
            ) as pool:
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

            # Restore original temperature
            if attempt > 1 and original_temperature == 0.0:
                self.metadata.llm.temperature = original_temperature

            logger.info(
                f"Attempt {attempt} complete: "
                f"{len(attempt_outputs)} instances processed"
            )
            all_outputs.extend(attempt_outputs)

            # If this is the last attempt, we're done
            if attempt == self.metadata.max_attempts:
                break

            # Evaluate which instances failed and need retry
            attempt_file = os.path.join(
                self.metadata.eval_output_dir, f"output.critic_attempt_{attempt}.jsonl"
            )

            failed_instance_ids = get_failed_instances(attempt_file, critic)

            # Filter instances for next attempt
            # Always include ALL failed instances from current attempt for next attempt
            instances_to_process = [
                inst for inst in all_instances if inst.id in failed_instance_ids
            ]

            logger.info(f"Found {len(failed_instance_ids)} failed instances for retry")

            if not instances_to_process:
                logger.info("All instances succeeded, stopping early")
                break

        # Aggregate results from all attempts
        logger.info("Aggregating results from all attempts")
        aggregate_results(
            output_dir=self.metadata.eval_output_dir,
            max_attempts=self.metadata.max_attempts,
            critic_name=critic_name,
            final_output_file="output.jsonl",
        )

        logger.info(
            f"Evaluation complete: {total_instances} total instances, "
            f"{self.metadata.max_attempts} max attempts"
        )
        return all_outputs

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
        logger.info("[child] start id=%s", instance.id)

        retry_count = 0
        last_error = None
        max_retries = self.metadata.max_retries

        while retry_count <= max_retries:
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

        # This should never be reached, but added for type safety
        error_output = self._create_error_output(
            instance, Exception("Unexpected error: no attempts made"), max_retries
        )
        return instance, error_output


# ---------- Optional per-process initializer ---------------------------------------


def _child_init() -> None:
    """Per-process initializer (placeholder).
    Put signal handlers or per-process setup here if needed.
    """
    pass
