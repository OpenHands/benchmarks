"""
Evaluation orchestrator.
"""

import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel, Field
from tqdm import tqdm

from benchmarks.utils.constants import OUTPUT_FILENAME
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import get_logger
from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)


def _validate_env() -> None:
    missing = [
        name for name in ("AGENT_SDK_PATH", "LLM_API_KEY") if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


OnResult = Callable[[EvalInstance, EvalOutput], None]


class Evaluation(ABC, BaseModel):
    """Abstract orchestrator for instance processing (process-based)."""

    metadata: EvalMetadata
    num_workers: int = Field(default=1, ge=1)

    @property
    def output_path(self) -> str:
        return os.path.join(self.metadata.eval_output_dir, OUTPUT_FILENAME)

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
        _validate_env()

        logger.info("Starting evaluation (process pool)")
        logger.info("metadata=%s", self.metadata)
        logger.info("workers=%d", self.num_workers)

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
