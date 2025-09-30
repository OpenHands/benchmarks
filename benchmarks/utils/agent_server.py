"""
Agent server utilities for remote runtime execution.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.run_evaluation import (
    create_workspace_for_instance,
    get_instruction,
    process_instance_simplified,
)
from benchmarks.utils.shared import EvalMetadata
from openhands.sdk import Conversation, get_logger
from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation
from openhands.tools.preset.default import get_default_agent


logger = get_logger(__name__)


def read_completed_instances(output_file: str) -> set:
    """Read completed instance IDs from existing output file."""
    completed_instances = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            result = json.loads(line)
                            if "instance_id" in result:
                                completed_instances.add(result["instance_id"])
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning(f"Error reading existing results from {output_file}: {e}")
    return completed_instances


def _stream_output(stream, prefix, target_stream):
    """Stream output from subprocess to target stream with prefix."""
    try:
        for line in iter(stream.readline, ""):
            if line:
                target_stream.write(f"[{prefix}] {line}")
                target_stream.flush()
    except Exception as e:
        print(f"Error streaming {prefix}: {e}", file=sys.stderr)
    finally:
        stream.close()


class ManagedAPIServer:
    """Context manager for subprocess-managed OpenHands API server."""

    def __init__(self, port: int = 8000, host: str = "127.0.0.1"):
        self.port = port
        self.host = host
        self.process = None
        self.base_url = f"http://{host}:{port}"
        self.stdout_thread = None
        self.stderr_thread = None

    def __enter__(self):
        """Start the API server subprocess."""
        print(f"Starting OpenHands API server on {self.base_url}...")

        # Start the server process
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "openhands.agent_server",
                "--port",
                str(self.port),
                "--host",
                self.host,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                "LOG_JSON": "true", 
                "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
                "PATH": os.environ.get("PATH", ""),
                **os.environ
            },
        )

        # Start threads to stream stdout and stderr
        self.stdout_thread = threading.Thread(
            target=_stream_output,
            args=(self.process.stdout, "SERVER", sys.stdout),
            daemon=True,
        )
        self.stderr_thread = threading.Thread(
            target=_stream_output,
            args=(self.process.stderr, "SERVER", sys.stderr),
            daemon=True,
        )

        self.stdout_thread.start()
        self.stderr_thread.start()

        # Wait for server to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                import httpx

                response = httpx.get(f"{self.base_url}/health", timeout=1.0)
                if response.status_code == 200:
                    print(f"API server is ready at {self.base_url}")
                    return self
            except Exception:
                pass

            if self.process.poll() is not None:
                # Process has terminated
                raise RuntimeError(
                    "Server process terminated unexpectedly. "
                    "Check the server logs above for details."
                )

            time.sleep(1)

        raise RuntimeError(f"Server failed to start after {max_retries} seconds")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the API server subprocess."""
        if self.process:
            print("Stopping API server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Force killing API server...")
                self.process.kill()
                self.process.wait()

            # Wait for streaming threads to finish (they're daemon threads,
            # so they'll stop automatically)
            # But give them a moment to flush any remaining output
            time.sleep(0.5)
            print("API server stopped.")


def run_remote_evaluation(llm: Any, metadata: EvalMetadata, num_workers: int = 1) -> None:
    """
    Run evaluation using remote runtime mode (agent server).
    
    Args:
        llm: LLM instance to use for evaluation
        metadata: EvalMetadata object containing evaluation configuration
        num_workers: Number of worker threads to use for parallel processing
    """
    logger.info("Running evaluation in REMOTE mode")
    logger.info(f"Using {num_workers} workers for parallel processing")
    
    # Import Runtime class
    from benchmarks.utils.runtime import Runtime
    
    # Global variables for runtime methods
    global instances, output_file, results, agent, llm_instance
    instances = None
    output_file = None
    results = []
    agent = None
    llm_instance = llm
    def initialize_runtime():
        """Initialize the runtime and return instances to process."""
        global instances, output_file, agent
        
        # Create agent
        agent = get_default_agent(
            llm=llm_instance,
            working_dir=str(Path.cwd()),
            cli_mode=True,  # Disable browser tools for simplicity
        )

        # Prepare output file
        output_file = os.path.join(metadata.eval_output_dir or ".", "output.jsonl")
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Read existing completed instances instead of overwriting
        completed_instances = read_completed_instances(output_file)
        if completed_instances:
            logger.info(f"Found {len(completed_instances)} already completed instances")
        else:
            logger.info("No existing results found, starting fresh")
            # Create empty output file only if it doesn't exist
            if not os.path.exists(output_file):
                with open(output_file, "w"):
                    pass

        # Retrieve instances to process, excluding completed ones
        instances = get_dataset(
            metadata.dataset or "",
            metadata.data_split or "",
            output_file,
            metadata.eval_n_limit or 0,
            completed_instances,
        )
        print(f"### OUTPUT FILE: {output_file} ###")
        return instances

    def process_instance(instance):
        """Process a single instance using remote conversation."""
        logger.info(f"Processing instance: {instance.instance_id}")

        # Create workspace and get actual path
        workspace_path = create_workspace_for_instance(instance, metadata)
        instruction = get_instruction(
            instance, metadata, workspace_path, metadata.prompt_path or ""
        )

        # Get the worker's server port (this will be set by the worker)
        worker_port = getattr(threading.current_thread(), 'server_port', 8001)
        server_url = f"http://localhost:{worker_port}"
        
        conversation = None
        
        try:
            # Create RemoteConversation (server should already be running from worker setup)
            conversation = Conversation(
                agent=agent,
                host=server_url,
                visualize=False,
                stuck_detection=False,  # Disable stuck detection to avoid FileNotFoundError in remote runtime
            )
            
            from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation
            assert isinstance(conversation, RemoteConversation)

            # Send message and run with event streaming
            logger.info(f"Sending instruction to conversation: {instruction[:100]}...")
            conversation.send_message(instruction)
            
            # Add callback to log events as they happen
            def log_event(event):
                event_type = type(event).__name__
                event_content = getattr(event, 'message', getattr(event, 'content', getattr(event, 'action', str(event))))
                logger.info(f"Event: {event_type} - {str(event_content)[:100]}")
            
            # Start WebSocket client for event streaming
            from openhands.sdk.conversation.impl.remote_conversation import WebSocketCallbackClient
            # Extract port from the conversation's client base URL
            base_url = str(conversation._client.base_url)
            ws_client = WebSocketCallbackClient(
                host=base_url,
                conversation_id=str(conversation._id),
                callbacks=[log_event]
            )
            ws_client.start()
            
            logger.info("Starting conversation.run()...")
            try:
                conversation.run()
                logger.info("Conversation.run() completed")
            finally:
                ws_client.stop()
            
            # Check conversation state
            logger.info(f"Conversation status: {conversation.state.agent_status}")
            logger.info(f"Number of events: {len(list(conversation.state.events))}")

            # Extract conversation history
            logger.info(f"DEBUG: Conversation type: {type(conversation)}")
            logger.info(f"DEBUG: Conversation state type: {type(conversation.state)}")
            logger.info(f"DEBUG: Has events attribute: {hasattr(conversation.state, 'events')}")
            
            try:
                # For remote conversations, try to force a sync first
                if hasattr(conversation.state, 'events') and hasattr(conversation.state.events, '_do_full_sync'):
                    logger.info("DEBUG: Forcing full sync for remote events...")
                    conversation.state.events._do_full_sync()
                
                history = list(conversation.state.events)
                logger.info(f"Extracted {len(history)} events from conversation history")
                
                # Log some details about the events
                if history:
                    logger.info(f"DEBUG: First event type: {type(history[0])}")
                    logger.info(f"DEBUG: Last event type: {type(history[-1])}")
                else:
                    logger.warning("DEBUG: No events found in conversation.state.events")
                    
            except Exception as e:
                logger.error(f"Error extracting conversation history: {e}")
                logger.info(f"DEBUG: Trying alternative history extraction methods...")
                
                # Try alternative methods to get conversation history
                history = []
                if hasattr(conversation, '_events'):
                    history = list(conversation._events)
                    logger.info(f"Found {len(history)} events in conversation._events")
                elif hasattr(conversation, 'events'):
                    history = list(conversation.events)
                    logger.info(f"Found {len(history)} events in conversation.events")
                elif hasattr(conversation.state, '_events'):
                    history = list(conversation.state._events)
                    logger.info(f"Found {len(history)} events in conversation.state._events")
                else:
                    logger.error("No events found in conversation object")
                    # Try to inspect the conversation object
                    logger.info(f"DEBUG: Conversation attributes: {dir(conversation)}")
                    logger.info(f"DEBUG: Conversation state attributes: {dir(conversation.state)}")
                    history = []

            # Extract git patch from conversation history
            git_patch = ""
            workspace_path = None
            
            try:
                # Look for workspace path and any git diff output in conversation events
                import re
                logger.info(f"DEBUG: Analyzing {len(history)} events for git patches...")
                
                for i, event in enumerate(history):
                    event_type = type(event).__name__
                    logger.info(f"DEBUG: Event {i}: {event_type}")
                    
                    # Check different event attributes for content
                    content_sources = []
                    if hasattr(event, 'content'):
                        content_sources.append(('content', event.content))
                    if hasattr(event, 'observation'):
                        content_sources.append(('observation', event.observation))
                    if hasattr(event, 'action') and hasattr(event.action, 'content'):
                        content_sources.append(('action.content', event.action.content))
                    if hasattr(event, 'action') and hasattr(event.action, 'command'):
                        content_sources.append(('action.command', event.action.command))
                    
                    for source_name, content in content_sources:
                        if isinstance(content, str) and content:
                            logger.info(f"DEBUG: Event {i} {source_name} length: {len(content)}")
                            
                            # Extract workspace path if not found yet
                            if workspace_path is None and '/tmp/tmp' in content:
                                match = re.search(r'/tmp/tmp\w+/\w+', content)
                                if match:
                                    workspace_path = match.group(0)
                                    logger.info(f"Found workspace path: {workspace_path}")
                            
                            # Look for git diff output in event content
                            if ('diff --git' in content or 
                                ('--- a/' in content and '+++ b/' in content) or
                                (content.startswith('diff ') and '@@' in content)):
                                git_patch = content
                                logger.info(f"Found git patch in {source_name}: {len(git_patch)} characters")
                                logger.info(f"Git patch preview: {git_patch[:200]}...")
                                break
                            
                            # Also look for git commands that might produce diffs
                            if 'git diff' in content.lower():
                                logger.info(f"Found 'git diff' command in {source_name}: {content[:100]}...")
                    
                    # Also check if event has action with path
                    if hasattr(event, 'action') and hasattr(event.action, 'path'):
                        if workspace_path is None and '/tmp/tmp' in str(event.action.path):
                            match = re.search(r'/tmp/tmp\w+/\w+', str(event.action.path))
                            if match:
                                workspace_path = match.group(0)
                                logger.info(f"Found workspace path from action: {workspace_path}")
                
                # If no git patch found in history but we have workspace path, 
                # assume there were no changes (empty patch)
                if not git_patch and workspace_path:
                    logger.info("No git patch found in conversation history - assuming no changes made")
                    git_patch = ""
                    
            except Exception as e:
                logger.error(f"Error extracting git patch: {e}")
                git_patch = ""

            logger.info(f"Extracted git patch with {len(git_patch)} characters")

            # Extract results from conversation state
            logger.info("Starting result extraction from conversation state")
            from benchmarks.utils.shared import EvalOutput
            
            logger.info(f"Creating EvalOutput with: instance_id={instance.instance_id}, history_events={len(history)}, git_patch_length={len(git_patch)}")
            
            result = EvalOutput(
                instance_id=instance.instance_id,
                instruction=instruction,
                test_result={
                    "git_patch": git_patch,
                    "resolved": len(git_patch) > 0,  # Consider resolved if there are changes
                },
                metadata=metadata.model_dump(),
                history=history,
                metrics={},
                error=None,
            )

            # Save result using the complete format
            result_dict = result.model_dump(mode="json")

            logger.info(f"Writing result for {instance.instance_id} to {output_file}")
            logger.info(f"Result dict keys: {list(result_dict.keys())}")
            logger.info(f"Result dict git_patch length: {len(result_dict.get('test_result', {}).get('git_patch', ''))}")
            logger.info(f"Result dict history length: {len(result_dict.get('history', []))}")

            # Write to output file (thread-safe)
            import json
            
            # Use a lock to ensure thread-safe file writing
            if not hasattr(process_instance, '_file_lock'):
                process_instance._file_lock = threading.Lock()
            
            with process_instance._file_lock:
                with open(output_file, "a") as f:
                    json_line = json.dumps(result_dict) + "\n"
                    f.write(json_line)
                    f.flush()  # Ensure it's written immediately
                    logger.info(
                        f"Successfully wrote {len(json_line)} characters to output file"
                    )

            logger.info(f"Completed processing instance {instance.instance_id}")
            return result

        except Exception as e:
            logger.error(f"Error processing instance {instance.instance_id}: {e}")
            raise  # Re-raise to let the worker handle it

        finally:
            # Clean up conversation
            if conversation:
                conversation.close()

    def complete_runtime():
        """Complete the runtime - any cleanup if needed."""
        logger.info("Remote evaluation completed!")

    # Create and run the Runtime
    runtime = Runtime(
        metadata=metadata,
        initialize_runtime=initialize_runtime,
        process_instance=process_instance,
        complete_runtime=complete_runtime,
        num_workers=num_workers,
    )

    runtime.run()