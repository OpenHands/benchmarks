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
from benchmarks.utils.iterative import (
    AgentFinishedCritic,
    aggregate_results,
    get_failed_instances,
)
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

    # --- Runner ---
    def run(
        self,
        *,
        on_result: Optional[OnResult] = None,
    ) -> List[EvalOutput]:
        """
        Run evaluation with iterative mode support.

        If max_attempts > 1, will retry failed instances multiple times.
        """
        logger.info("Starting evaluation (process pool)")
        logger.info("metadata=%s", self.metadata)
        logger.info("workers=%d", self.num_workers)
        logger.info("max_attempts=%d", self.metadata.max_attempts)

        if self.metadata.max_attempts == 1:
            # Single attempt mode - use original logic
            return self._run_single_attempt(on_result=on_result)
        else:
            # Iterative mode - multiple attempts
            return self._run_iterative_mode(on_result=on_result)

    def _run_single_attempt(
        self,
        *,
        on_result: Optional[OnResult] = None,
    ) -> List[EvalOutput]:
        """Run evaluation with single attempt (original behavior)."""
        instances = self.prepare_instances()
        total = len(instances)
        logger.info("prepared %d instances", total)
        if total == 0:
            logger.warning("No instances to process.")
            return []

        outputs: List[EvalOutput] = []

        # Submit one process per instance. Using a *bound* instance method as
        # the target; this requires the subclass to be importable/pickleable
        # (top-level class, no closures).
        with ProcessPoolExecutor(
            max_workers=self.num_workers, initializer=_child_init
        ) as pool:
            futures = [pool.submit(self._process_one_mp, inst) for inst in instances]

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Evaluating",
                leave=False,
            ):
                try:
                    instance, out = fut.result()
                except Exception as e:
                    logger.error(
                        f"Error during instance evaluation: {e}",
                        exc_info=True,
                        stack_info=True,
                    )
                    raise

                outputs.append(out)
                if on_result:
                    try:
                        on_result(instance, out)
                    except Exception as cb_err:
                        logger.warning("on_result callback failed: %s", cb_err)

        logger.info("Evaluation complete: %d/%d done", len(outputs), total)
        return outputs

    def _run_iterative_mode(
        self,
        *,
        on_result: Optional[OnResult] = None,
    ) -> List[EvalOutput]:
        """Run evaluation with iterative mode (multiple attempts)."""
        # Get all instances for the first attempt
        all_instances = self.prepare_instances()
        total_instances = len(all_instances)
        logger.info("prepared %d instances for iterative evaluation", total_instances)

        if total_instances == 0:
            logger.warning("No instances to process.")
            return []

        critic = AgentFinishedCritic()
        all_outputs: List[EvalOutput] = []

        # Track instances to process in each attempt
        instances_to_process = all_instances.copy()

        for attempt in range(1, self.metadata.max_attempts + 1):
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
                    except Exception as e:
                        logger.error(
                            f"Error during instance evaluation: {e}",
                            exc_info=True,
                            stack_info=True,
                        )
                        raise

                    attempt_on_result(instance, out)

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
            instances_to_process = [
                inst for inst in instances_to_process if inst.id in failed_instance_ids
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
            final_output_file="output.jsonl",
        )

        logger.info(
            f"Iterative evaluation complete: {total_instances} total instances, "
            f"{self.metadata.max_attempts} max attempts"
        )
        return all_outputs

    # --- Worker-side method (executed in child processes) ---------------------------
    def _process_one_mp(
        self, instance: EvalInstance
    ) -> Tuple[EvalInstance, EvalOutput]:
        """Execute one instance in a child process.

        - Creates workspace in the *child* process
        - Ensures proper context-managed cleanup
        - Returns (instance, output) so the parent can stream results
        """
        logger.info("[child] start id=%s", instance.id)

        workspace = self.prepare_workspace(instance)
        out = self.evaluate_instance(instance, workspace)
        logger.info("[child] done id=%s", instance.id)
        return instance, out


# ---------- Optional per-process initializer ---------------------------------------


def _child_init() -> None:
    """Per-process initializer (placeholder).
    Put signal handlers or per-process setup here if needed.
    """
    pass
