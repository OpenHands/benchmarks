"""
Evaluation class for orchestrating instance processing workflows.
"""

from __future__ import annotations

import os
import queue
import socket
import threading
from typing import Any, Callable

import pandas as pd

from benchmarks.utils.shared import EvalMetadata
from openhands.sdk import get_logger
from openhands.workspace import DockerWorkspace


logger = get_logger(__name__)


class Evaluation:
    """
    Evaluation class that orchestrates the processing of instances.

    This class receives metadata and callback methods for processing instances,
    and provides a run method that coordinates the entire workflow.
    """

    def __init__(
        self,
        metadata: EvalMetadata,
        initialize_dataset_run: Callable[[], pd.DataFrame],
        process_instance: Callable[[Any, Any], None],
        complete_dataset_run: Callable[[], None],
        get_instance_docker_image: Callable[[Any], str],
        num_workers: int = 1,
    ):
        """
        Initialize the Evaluation with metadata and processing methods.

        Args:
            metadata: EvalMetadata object containing evaluation metadata
            initialize_dataset_run: Function to initialize dataset and return instances
            process_instance(instance, workspace): Function to process each instance
            complete_dataset_run: Function to complete the dataset evaluation
            num_workers: Number of worker threads to use for parallel processing
            get_instance_docker_image: Function to get Docker image for each instance
        """
        # Check required environment variables
        if not os.getenv("AGENT_SDK_PATH"):
            raise RuntimeError(
                "AGENT_SDK_PATH environment variable is required but not set. "
                "Please set it to the path of your OpenHands Agent SDK installation."
            )
        if not os.getenv("LLM_API_KEY"):
            raise RuntimeError(
                "LLM_API_KEY environment variable is required but not set."
            )

        self.metadata = metadata
        self.initialize_dataset_run = initialize_dataset_run
        self.process_instance = process_instance
        self.complete_dataset_run = complete_dataset_run
        self.num_workers = num_workers
        self.get_instance_docker_image = get_instance_docker_image

        # Worker pool management
        self.instance_queue = queue.Queue()
        self.workers = []

        # Progress tracking
        self.total_instances = 0
        self.completed_count = 0
        self.progress_lock = threading.Lock()

    def _print_progress_bar(self) -> None:
        """Print a progress bar showing completion status."""
        if self.total_instances == 0:
            return

        percentage = (self.completed_count / self.total_instances) * 100
        bar_length = 50
        filled_length = int(bar_length * self.completed_count // self.total_instances)

        bar = "█" * filled_length + "▌" * (
            1
            if filled_length < bar_length
            and self.completed_count < self.total_instances
            else 0
        )
        bar = bar.ljust(bar_length)

        print(
            f"\r{percentage:.1f}%|{bar}| {self.completed_count}/{self.total_instances}",
            end="",
            flush=True,
        )

    def _worker_loop(self, worker_id: int) -> None:
        """
        Worker loop that processes instances from the queue.

        Args:
            worker_id: Unique identifier for this worker
        """
        logger.info(f"Worker {worker_id} starting...")

        # Start workspace
        agent_server = None

        try:
            while True:
                try:
                    # Get instance from queue with timeout
                    instance = self.instance_queue.get(timeout=1)
                    instance_id = instance.instance_id

                    logger.info(f"Worker {worker_id} processing instance {instance_id}")

                    # Get fresh port and start a per-instance workspace
                    instance_workspace = None
                    server_port = None
                    server_port = Evaluation._find_free_port(8001)
                    setattr(threading.current_thread(), "server_port", server_port)
                    logger.info(
                        (
                            f"Worker {worker_id} starting workspace"
                            f"for {instance_id} on port {server_port}"
                        )
                    )
                    try:
                        logger.info("Calling _start_workspace_for_instance...")
                        instance_workspace = self._start_workspace_for_instance(
                            instance, server_port
                        )
                        logger.info(f"Worker {worker_id} started for {instance_id}")
                    except Exception as e:
                        logger.error(
                            f"Worker {worker_id} failed to start {instance_id}"
                        )
                        logger.error(
                            f"Full exception details: {type(e).__name__}: {str(e)}"
                        )
                        import traceback

                        logger.error(f"Traceback: {traceback.format_exc()}")
                        self.instance_queue.task_done()
                        continue

                    try:
                        # Process the instance using the provided callback
                        self.process_instance(instance, instance_workspace)
                        logger.info(f"Worker {worker_id} completed {instance_id}")
                    except Exception as e:
                        logger.error(f"Worker {worker_id} error processing instance")
                        logger.error(f"{instance_id}: {e}")
                    finally:
                        # Cleanup per-instance workspace
                        if instance_workspace:
                            try:
                                instance_workspace.__exit__(None, None, None)
                                logger.info(f"Worker {worker_id} stopped {instance_id}")
                            except Exception as e:
                                logger.error(f"Error stopping {worker_id}")
                                logger.error(f"{instance_id}: {e}")

                        # Mark task as done
                        self.instance_queue.task_done()

                        # Update progress
                        with self.progress_lock:
                            self.completed_count += 1
                            self._print_progress_bar()

                        # Clean up thread port
                        if hasattr(threading.current_thread(), "server_port"):
                            delattr(threading.current_thread(), "server_port")

                except queue.Empty:
                    # No more instances to process
                    break

        finally:
            # Cleanup: Stop workspace for this worker
            if agent_server:
                try:
                    agent_server.__exit__(
                        None, None, None
                    )  # Stop the workspace using context manager protocol
                    logger.info(f"Worker {worker_id} stopped workspace")
                except Exception as e:
                    logger.error(f"Worker {worker_id} error stopping workspace: {e}")

        logger.info(f"Worker {worker_id} finished")

    def _start_workspace_for_instance(self, instance: Any, base_port: int):
        """
        Start a Docker workspace for a specific instance.

        Args:
            instance: The instance data
            base_port: Base port number for the workspace

        Returns:
            DockerWorkspace instance
        """
        logger.info(
            f"Start workspace instance={getattr(instance, 'instance_id', 'unknown')}"
        )
        logger.info(f"Start workspace with base_port={base_port}")

        # Get the Docker image for this instance
        docker_image = self.get_instance_docker_image(instance)
        logger.info(f"Got docker_image={docker_image} for instance")

        # Create and start DockerWorkspace
        logger.info(f"Creating DockerWorkspace with base_image={docker_image}")
        logger.info(f"Creating DockerWorkspace with host_port={base_port}")
        workspace = DockerWorkspace(
            base_image=docker_image,
            host_port=base_port,
            forward_env=["LLM_API_KEY"],
        )
        logger.info("DockerWorkspace created successfully")

        # Start the workspace using context manager protocol
        logger.info("Calling workspace.__enter__() to start the Docker container...")
        try:
            workspace.__enter__()
            logger.info("workspace.__enter__() completed successfully!")
        except Exception as e:
            logger.error(f"workspace.__enter__() failed: {type(e).__name__}: {str(e)}")
            import traceback

            logger.error(f"workspace.__enter__() traceback: {traceback.format_exc()}")
            raise

        return workspace

    @staticmethod
    def _find_free_port(start_port=8001):
        """Find the first available port starting from start_port."""
        port = start_port
        logger.info("start_port type=%s value=%s", type(start_port), start_port)
        while port < start_port + 1000:  # Try 100 ports max
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("localhost", port))
                    return port
            except OSError:
                port += 1
        raise RuntimeError(
            f"No free ports found in range {start_port}-{start_port + 99}"
        )

    def _start_workers(self) -> list[threading.Thread]:
        """
        Start worker threads for parallel processing.

        Returns:
            List of worker threads
        """
        workers = []

        for worker_id in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),  # Remove server_port parameter
                name=f"Worker-{worker_id}",
            )

            workers.append(worker)
            worker.start()
            logger.info(f"Started worker {worker_id}")

        return workers

    def run(self) -> None:
        """
        Run the complete instance processing workflow using worker pool.

        This method:
        1. Initializes the evaluation and retrieves all instances to process
        2. Starts worker threads for parallel processing
        3. Distributes instances to workers via queue
        4. Waits for all workers to complete
        5. Completes the evaluation
        """
        logger.info("Starting evaluation execution")
        logger.info(f"Evaluation metadata: {self.metadata}")
        logger.info(f"Using {self.num_workers} workers")

        try:
            # Initialize the evaluation and retrieve all instances to process
            logger.info("Initializing evaluation and retrieving instances")
            instances = self.initialize_dataset_run()
            logger.info(f"Retrieved {len(instances)} instances to process")

            # Initialize progress tracking
            self.total_instances = len(instances)
            self.completed_count = 0

            # Always use worker pool processing (even for single worker)
            logger.info(f"Using parallel processing with {self.num_workers} workers")

            # Add all instances to the queue
            for _, instance in instances.iterrows():
                self.instance_queue.put(instance)

            # Start worker threads
            workers = self._start_workers()

            # Wait for all instances to be processed
            logger.info("Waiting for all instances to be processed...")
            self.instance_queue.join()

            # Wait for all workers to finish
            logger.info("Waiting for all workers to finish...")
            for worker in workers:
                worker.join()

            logger.info("All workers completed")

        finally:
            # Always complete the evaluation, even if there were errors
            if self.total_instances > 0:
                print()  # Add newline after progress bar
            logger.info("Completing evaluation")
            try:
                self.complete_dataset_run()
            except Exception as e:
                logger.error(f"Error completing evaluation: {e}")

        logger.info("Evaluation execution completed")
