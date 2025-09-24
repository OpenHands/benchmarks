"""
Runtime class for orchestrating instance processing workflows.
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Any, Callable

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
        num_workers: int = 1,
    ):
        """
        Initialize the Runtime with metadata and processing methods.

        Args:
            metadata: EvalMetadata object containing runtime metadata
            initialize_runtime: Function to initialize the runtime and return instances
            process_instance: Function to process each instance
            complete_runtime: Function to complete the runtime (called once at end)
            num_workers: Number of worker threads to use for parallel processing
        """
        self.metadata = metadata
        self.initialize_runtime = initialize_runtime
        self.process_instance = process_instance
        self.complete_runtime = complete_runtime
        self.num_workers = num_workers
        
        # Worker pool management
        self.instance_queue = queue.Queue()
        self.workers = []
        self.is_remote_mode = os.getenv("RUNTIME", "local").lower() == "remote"

    def _worker_loop(self, worker_id: int, server_port: int = None) -> None:
        """
        Worker loop that processes instances from the queue.
        
        Args:
            worker_id: Unique identifier for this worker
            server_port: Port for agent server (remote mode only)
        """
        logger.info(f"Worker {worker_id} starting (remote_mode={self.is_remote_mode}, port={server_port})")
        
        # Set server_port on current thread for remote mode
        if self.is_remote_mode and server_port:
            threading.current_thread().server_port = server_port
        
        # For remote mode, we need to import and start the agent server
        agent_server = None
        if self.is_remote_mode and server_port:
            try:
                from benchmarks.utils.agent_server import ManagedAPIServer
                agent_server = ManagedAPIServer(port=server_port)
                agent_server.__enter__()  # Start the server using context manager protocol
                logger.info(f"Worker {worker_id} started agent server on port {server_port}")
            except Exception as e:
                logger.error(f"Worker {worker_id} failed to start agent server: {e}")
                return
        
        try:
            while True:
                try:
                    # Get instance from queue with timeout
                    instance = self.instance_queue.get(timeout=1)
                    
                    logger.info(f"Worker {worker_id} processing instance {instance.instance_id}")
                    
                    try:
                        # Process the instance using the provided callback
                        self.process_instance(instance)
                        logger.info(f"Worker {worker_id} completed instance {instance.instance_id}")
                    except Exception as e:
                        logger.error(f"Worker {worker_id} error processing instance {instance.instance_id}: {e}")
                    finally:
                        # Mark task as done
                        self.instance_queue.task_done()
                        
                except queue.Empty:
                    # No more instances to process
                    break
                    
        finally:
            # Cleanup: Stop agent server for this worker
            if agent_server:
                try:
                    agent_server.__exit__(None, None, None)  # Stop the server using context manager protocol
                    logger.info(f"Worker {worker_id} stopped agent server")
                except Exception as e:
                    logger.error(f"Worker {worker_id} error stopping agent server: {e}")
        
        logger.info(f"Worker {worker_id} finished")

    def _start_workers(self) -> list[threading.Thread]:
        """
        Start worker threads for parallel processing.
        
        Returns:
            List of worker threads
        """
        workers = []
        
        for worker_id in range(self.num_workers):
            if self.is_remote_mode:
                # Each worker gets its own agent server port
                server_port = 8001 + worker_id
                worker = threading.Thread(
                    target=self._worker_loop,
                    args=(worker_id, server_port),
                    name=f"Worker-{worker_id}"
                )
            else:
                # Local mode worker
                worker = threading.Thread(
                    target=self._worker_loop,
                    args=(worker_id, None),
                    name=f"Worker-{worker_id}"
                )
            
            workers.append(worker)
            worker.start()
            logger.info(f"Started worker {worker_id}")
        
        return workers

    def run(self) -> None:
        """
        Run the complete instance processing workflow using worker pool.

        This method:
        1. Initializes the runtime and retrieves all instances to process
        2. Starts worker threads for parallel processing
        3. Distributes instances to workers via queue
        4. Waits for all workers to complete
        5. Completes the runtime
        """
        logger.info("Starting runtime execution")
        logger.info(f"Runtime metadata: {self.metadata}")
        logger.info(f"Using {self.num_workers} workers in {'remote' if self.is_remote_mode else 'local'} mode")

        try:
            # Initialize the runtime and retrieve all instances to process
            logger.info("Initializing runtime and retrieving instances")
            instances = self.initialize_runtime()
            logger.info(f"Retrieved {len(instances)} instances to process")

            if self.num_workers == 1:
                # Single worker mode - use original sequential processing
                logger.info("Using sequential processing (single worker)")
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
            else:
                # Multi-worker mode - use parallel processing
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
            # Always complete the runtime, even if there were errors
            logger.info("Completing runtime")
            try:
                self.complete_runtime()
            except Exception as e:
                logger.error(f"Error completing runtime: {e}")

        logger.info("Runtime execution completed")
