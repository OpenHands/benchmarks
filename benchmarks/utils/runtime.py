"""
Runtime class for orchestrating instance processing workflows.
"""

from __future__ import annotations

from typing import Any, Callable, List
import pandas as pd

from benchmarks.utils.shared import EvalMetadata
from openhands.sdk import get_logger


logger = get_logger(__name__)


class Runtime:
    """
    Runtime class that orchestrates the processing of instances.

    This class receives metadata and callback methods for processing instances,
    and provides a run method that coordinates the entire workflow.
    """

    def __init__(
        self,
        metadata: EvalMetadata,
        initialize_runtime: Callable[[], pd.DataFrame],
        process_instance: Callable[[Any], None],
        complete_runtime: Callable[[], None],
    ):
        """
        Initialize the Runtime with metadata and processing methods.

        Args:
            metadata: EvalMetadata object containing runtime metadata
            initialize_runtime: Function to initialize the runtime and return instances
            process_instance: Function to process each instance
            complete_runtime: Function to complete the runtime (called once at end)
        """
        self.metadata = metadata
        self.initialize_runtime = initialize_runtime
        self.process_instance = process_instance
        self.complete_runtime = complete_runtime

    def run(self) -> None:
        """
        Run the complete instance processing workflow.

        This method:
        1. Initializes the runtime and retrieves all instances to process
        2. Processes each instance
        3. Completes the runtime
        """
        logger.info("Starting runtime execution")
        logger.info(f"Runtime metadata: {self.metadata}")

        try:
            # Initialize the runtime and retrieve all instances to process
            logger.info("Initializing runtime and retrieving instances")
            instances = self.initialize_runtime()
            logger.info(f"Retrieved {len(instances)} instances to process")

            # Process each instance
            for i, (_, instance) in enumerate(instances.iterrows()):
                logger.info(f"Processing instance {i + 1}/{len(instances)}")

                try:
                    # Process the instance
                    self.process_instance(instance)
                    logger.info(
                        f"Successfully completed instance {i + 1}/{len(instances)}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error processing instance {i + 1}/{len(instances)}: {e}"
                    )
                    # Continue with next instance rather than failing completely
                    continue

        finally:
            # Always complete the runtime, even if there were errors
            logger.info("Completing runtime")
            try:
                self.complete_runtime()
            except Exception as e:
                logger.error(f"Error completing runtime: {e}")

        logger.info("Runtime execution completed")
