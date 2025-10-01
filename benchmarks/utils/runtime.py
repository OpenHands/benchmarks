"""
Runtime class for orchestrating instance processing workflows.
"""

from __future__ import annotations

import os
import platform
import queue
import threading
import socket
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
        get_instance_docker_image: Callable[[Any], str] = None,
    ):
        """
        Initialize the Runtime with metadata and processing methods.

        Args:
            metadata: EvalMetadata object containing runtime metadata
            initialize_runtime: Function to initialize the runtime and return instances
            process_instance: Function to process each instance
            complete_runtime: Function to complete the runtime (called once at end)
            num_workers: Number of worker threads to use for parallel processing
            get_instance_docker_image: Function to get Docker image for each instance (remote mode only)
        """
        self.metadata = metadata
        self.initialize_runtime = initialize_runtime
        self.process_instance = process_instance
        self.complete_runtime = complete_runtime
        self.num_workers = num_workers
        self._get_instance_docker_image = get_instance_docker_image
        
        # Runtime mode detection
        self.runtime_mode = self._detect_runtime_mode()
        self.is_sandbox_mode = self.runtime_mode == 'remote'
        logger.info(f"DEBUG: Runtime initialized with runtime_mode='{self.runtime_mode}', is_sandbox_mode={self.is_sandbox_mode}")
        
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
        
        bar = '█' * filled_length + '▌' * (1 if filled_length < bar_length and self.completed_count < self.total_instances else 0)
        bar = bar.ljust(bar_length)
        
        print(f"\rProgress: {percentage:.1f}%|{bar}| {self.completed_count}/{self.total_instances}", end='', flush=True)

    def _detect_runtime_mode(self) -> str:
        """Detect runtime mode based on environment and configuration."""
        runtime_env = os.getenv("RUNTIME", "remote")
        logger.info(f"DEBUG: RUNTIME environment variable = '{runtime_env}'")
        if runtime_env.lower() == "remote":
            logger.info(f"DEBUG: Detected runtime mode: 'remote' (sandbox mode)")
            return 'remote'  # RUNTIME=remote always means sandboxed mode
        else:
            logger.error(f"DEBUG: Local runtime mode is not supported")
            raise ValueError(f"Local runtime mode is not supported. RUNTIME environment variable is set to '{runtime_env}', but only 'remote' mode is allowed.")

    def get_instance_docker_image(self, instance: Any) -> str:
        """
        Get the Docker image to use for a specific instance.
        
        Args:
            instance: The instance data as a pandas Series
            
        Returns:
            Docker image name to use for this instance
        """
        if self._get_instance_docker_image:
            return self._get_instance_docker_image(instance)
        
        # Default fallback logic
        return self._get_default_docker_image(instance)
    
    def _get_default_docker_image(self, instance: Any) -> str:
        """Default logic for determining Docker image based on instance."""
        # Check if instance has specific requirements
        if hasattr(instance, 'language') or (hasattr(instance, '__contains__') and 'language' in instance):
            language = getattr(instance, 'language', instance.get('language', '') if hasattr(instance, 'get') else '').lower()
            if language == 'python':
                return "python:3.12-slim"
            elif language == 'javascript' or language == 'node':
                return "node:18-slim"
            elif language == 'java':
                return "openjdk:17-slim"
        
        # Check for specific frameworks or dependencies
        if hasattr(instance, 'repo_name') or (hasattr(instance, '__contains__') and 'repo_name' in instance):
            repo_name = getattr(instance, 'repo_name', instance.get('repo_name', '') if hasattr(instance, 'get') else '').lower()
            if 'django' in repo_name or 'flask' in repo_name:
                return "python:3.12-slim"
            elif 'react' in repo_name or 'vue' in repo_name:
                return "nikolaik/python-nodejs:python3.12-nodejs22"
        
        # Check if instance specifies its own Docker image
        if hasattr(instance, 'docker_image') and instance.docker_image:
            return instance.docker_image
        
        if hasattr(instance, '__contains__') and 'docker_image' in instance and instance['docker_image']:
            return instance['docker_image']
        
        # Default general-purpose image
        return os.getenv("SANDBOX_BASE_IMAGE", "nikolaik/python-nodejs:python3.12-nodejs22")

    def _worker_loop(self, worker_id: int, server_port: int = None) -> None:
        """
        Worker loop that processes instances from the queue.
        
        Args:
            worker_id: Unique identifier for this worker
            server_port: Port for agent server (remote mode only)
        """
        logger.info(f"Worker {worker_id} starting (mode={self.runtime_mode}, port={server_port})")
        
        # Set server_port on current thread for remote mode
        if self.is_sandbox_mode and server_port:
            threading.current_thread().server_port = server_port
        
        # Start appropriate agent server based on runtime mode
        agent_server = None
        if self.is_sandbox_mode and server_port:
            try:
                from openhands.sdk.sandbox import DockerSandboxedAgentServer
                # For remote mode, we'll start servers per instance in the processing loop
                logger.info(f"Worker {worker_id} ready for remote mode on port {server_port}")
            except ImportError as e:
                logger.error(f"Worker {worker_id} failed to import DockerSandboxedAgentServer: {e}")
                return
        
        try:
            while True:
                try:
                    # Get instance from queue with timeout
                    instance = self.instance_queue.get(timeout=1)
                    
                    logger.info(f"Worker {worker_id} processing instance {instance.instance_id}")
                    
                    # For sandbox mode, start a per-instance server
                    instance_server = None
                    if self.is_sandbox_mode and server_port:
                        logger.info(f"DEBUG: Worker {worker_id} attempting to start sandbox server for instance {instance.instance_id} on port {server_port}")
                        logger.info(f"DEBUG: is_sandbox_mode={self.is_sandbox_mode}, server_port={server_port}")
                        try:
                            logger.info(f"DEBUG: Calling _start_sandbox_server_for_instance...")
                            instance_server = self._start_sandbox_server_for_instance(instance, server_port)
                            logger.info(f"Worker {worker_id} started sandbox server for instance {instance.instance_id}")
                        except Exception as e:
                            logger.error(f"Worker {worker_id} failed to start sandbox server for instance {instance.instance_id}: {e}")
                            logger.error(f"DEBUG: Full exception details: {type(e).__name__}: {str(e)}")
                            import traceback
                            logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
                            continue
                    
                    try:
                        # Process the instance using the provided callback
                        self.process_instance(instance)
                        logger.info(f"Worker {worker_id} completed instance {instance.instance_id}")
                    except Exception as e:
                        logger.error(f"Worker {worker_id} error processing instance {instance.instance_id}: {e}")
                    finally:
                        # Cleanup per-instance sandbox server
                        if instance_server:
                            try:
                                instance_server.__exit__(None, None, None)
                                logger.info(f"Worker {worker_id} stopped sandbox server for instance {instance.instance_id}")
                            except Exception as e:
                                logger.error(f"Worker {worker_id} error stopping sandbox server for instance {instance.instance_id}: {e}")
                        
                        # Mark task as done
                        self.instance_queue.task_done()
                        
                        # Update progress
                        with self.progress_lock:
                            self.completed_count += 1
                            self._print_progress_bar()
                        
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

    def _start_sandbox_server_for_instance(self, instance: Any, base_port: int):
        """
        Start a sandboxed agent server for a specific instance.
        
        Args:
            instance: The instance data
            base_port: Base port number for the server
            
        Returns:
            DockerSandboxedAgentServer instance
        """
        logger.info(f"DEBUG: _start_sandbox_server_for_instance called with instance={getattr(instance, 'instance_id', 'unknown')}, base_port={base_port}")
        
        from openhands.sdk.sandbox import DockerSandboxedAgentServer
        logger.info(f"DEBUG: Successfully imported DockerSandboxedAgentServer")
        
        # Get the Docker image for this instance
        docker_image = self.get_instance_docker_image(instance)
        logger.info(f"DEBUG: Got docker_image={docker_image} for instance")
        
        # Note: DockerSandboxedAgentServer will create its own container name
        
        # Start the sandboxed server with the evaluation image
        # This builds the OpenHands agent server on top of the evaluation environment
        # Note: This requires the OpenHands source code with build.sh script
        # TODO: Either provide OpenHands source or use pre-built evaluation+agent images
        logger.info(f"DEBUG: Creating DockerSandboxedAgentServer with base_image={docker_image}, host_port={base_port}")
        server = DockerSandboxedAgentServer(
            base_image=docker_image,
            host_port=base_port,
        )
        logger.info(f"DEBUG: DockerSandboxedAgentServer created successfully")
        
        # Start the server using context manager protocol
        logger.info(f"DEBUG: Calling server.__enter__() to start the Docker container...")
        try:
            server.__enter__()
            logger.info(f"DEBUG: server.__enter__() completed successfully!")
        except Exception as e:
            logger.error(f"DEBUG: server.__enter__() failed: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"DEBUG: server.__enter__() traceback: {traceback.format_exc()}")
            raise
        
        return server

    @staticmethod
    def _find_free_port(start_port=8001):
        """Find the first available port starting from start_port."""
        port = start_port
        logger.info("start_port type=%s value=%s", type(start_port), start_port)
        while port < start_port + 100:  # Try 100 ports max
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                port += 1
        raise RuntimeError(f"No free ports found in range {start_port}-{start_port + 99}")

    def _start_workers(self) -> list[threading.Thread]:
        """
        Start worker threads for parallel processing.
        
        Returns:
            List of worker threads
        """
        workers = []
        
        for worker_id in range(self.num_workers):
            if self.is_sandbox_mode:
                # Each worker gets its own agent server port
                server_port = Runtime._find_free_port(8001 + worker_id*10)
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
        logger.info(f"Using {self.num_workers} workers in {'remote' if self.is_sandbox_mode else 'local'} mode")

        try:
            # Initialize the runtime and retrieve all instances to process
            logger.info("Initializing runtime and retrieving instances")
            instances = self.initialize_runtime()
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
            # Always complete the runtime, even if there were errors
            if self.total_instances > 0:
                print()  # Add newline after progress bar
            logger.info("Completing runtime")
            try:
                self.complete_runtime()
            except Exception as e:
                logger.error(f"Error completing runtime: {e}")

        logger.info("Runtime execution completed")
