"""
Agent server utilities for remote runtime execution.
"""

from __future__ import annotations

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
from openhands.sdk.preset.default import get_default_agent


logger = get_logger(__name__)


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


def run_remote_evaluation(llm: Any, metadata: EvalMetadata) -> None:
    """
    Run evaluation using remote runtime mode (agent server).
    
    Args:
        llm: LLM instance to use for evaluation
        metadata: EvalMetadata object containing evaluation configuration
    """
    logger.info("Running evaluation in REMOTE mode")
    
    # Use managed API server
    with ManagedAPIServer(port=8001) as server:
        # Create agent
        agent = get_default_agent(
            llm=llm,
            working_dir=str(Path.cwd()),
            cli_mode=True,  # Disable browser tools for simplicity
        )

        # Prepare output file
        output_file = os.path.join(metadata.eval_output_dir or ".", "output.jsonl")
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Create empty output file
        with open(output_file, "w"):
            pass

        # Retrieve instances to process
        instances = get_dataset(
            metadata.dataset or "",
            metadata.data_split or "",
            output_file,
            metadata.eval_n_limit or 0,
        )
        print(f"### OUTPUT FILE: {output_file} ###")

        # Process each instance using remote conversation
        for i, (_, instance) in enumerate(instances.iterrows()):
            logger.info(f"Processing instance {i + 1}/{len(instances)}: {instance.instance_id}")

            # Create workspace and get actual path
            workspace_path = create_workspace_for_instance(instance, metadata)
            instruction = get_instruction(
                instance, metadata, workspace_path, metadata.prompt_path or ""
            )

            # Create RemoteConversation
            conversation = Conversation(
                agent=agent,
                host=server.base_url,
                visualize=False,
            )
            assert isinstance(conversation, RemoteConversation)

            try:
                # Send message and run
                conversation.send_message(instruction)
                conversation.run()

                # Process the result
                result = process_instance_simplified(instance, instruction, metadata, workspace_path)

                # Save result using the complete format
                result_dict = result.model_dump(mode="json")
                if result.error:
                    result_dict["error"] = result.error

                logger.info(f"Writing result for {instance.instance_id} to {output_file}")
                logger.info(f"Result dict keys: {list(result_dict.keys())}")
                git_patch_len = len(result_dict.get("test_result", {}).get("git_patch", ""))
                logger.info(f"Git patch length: {git_patch_len}")

                # Write to output file
                import json
                with open(output_file, "a") as f:
                    json_line = json.dumps(result_dict) + "\n"
                    f.write(json_line)
                    f.flush()  # Ensure it's written immediately
                    logger.info(
                        f"Successfully wrote {len(json_line)} characters to output file"
                    )

            except Exception as e:
                logger.error(f"Error processing instance {instance.instance_id}: {e}")
                # Continue with next instance rather than failing completely
                continue

            finally:
                # Clean up conversation
                conversation.close()

    logger.info("Remote evaluation completed!")