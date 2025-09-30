from __future__ import annotations

import os
import threading

from pydantic import SecretStr

from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.run_evaluation import (
    construct_eval_output_dir,
    make_metadata,
)
from openhands.sdk import LLM, get_logger
from openhands.tools.preset.default import get_default_agent
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.run_evaluation import (
    get_instruction,
    read_completed_instances,
    write_output_to_file
)
from benchmarks.utils.conversation_tools import get_history, get_git_patch_from_history
from benchmarks.utils.runtime import Runtime
from openhands.sdk import Conversation, get_logger
from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation


logger = get_logger(__name__)


def create_runtime(llm: Any, metadata: EvalMetadata, num_workers: int = 1) -> Runtime:
    """
    Run evaluation using remote runtime mode (agent server).
    
    Args:
        llm: LLM instance to use for evaluation
        metadata: EvalMetadata object containing evaluation configuration
        num_workers: Number of worker threads to use for parallel processing
    """
    logger.info("Running evaluation in REMOTE mode")
    logger.info(f"Using {num_workers} workers for parallel processing")
    
    
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

        workspace_path = "/workspace"
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
                max_iteration_per_run=metadata.max_iterations,
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
            
            history = get_history(conversation)
            git_patch = get_git_patch_from_history(history)
            
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
                },
                metadata=metadata.model_dump(),
                history=history,
                metrics={},
                error=None,
            )

            write_output_to_file(instance, process_instance, result, output_file)

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

    return runtime

def main():
    default_prompt_path = os.path.join(
        os.path.dirname(__file__), "prompts", "default.j2"
    )
    parser = get_parser()
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=default_prompt_path,
        help="Path to prompt template file",
    )
    args = parser.parse_args()

    DATASET = args.dataset
    SPLIT = args.split
    MODEL = args.llm_config
    EVAL_OUTPUT_DIR = args.eval_output_dir
    MAX_ITERATIONS = args.max_iterations
    EVAL_N_LIMIT = args.eval_n_limit
    EVAL_NOTE = args.eval_note
    PROMPT_PATH = args.prompt_path

    # Create LLM instance
    api_key = os.getenv("LITELLM_API_KEY")
    if not api_key:
        raise ValueError("LITELLM_API_KEY environment variable is not set")
    llm = LLM(
        model=MODEL,
        api_key=SecretStr(api_key),
        base_url="https://llm-proxy.eval.all-hands.dev",
        temperature=0,
        service_id="litellm_proxy",
    )

    dataset_description = DATASET.replace("/", "__") + "-" + SPLIT.replace("/", "__")

    # Construct proper structured output directory path
    structured_output_dir = construct_eval_output_dir(
        base_dir=EVAL_OUTPUT_DIR,
        dataset_name=dataset_description,
        model=llm.model,
        max_iterations=MAX_ITERATIONS,
        eval_note=EVAL_NOTE,
    )

    metadata = make_metadata(
        llm,
        dataset_description,
        MAX_ITERATIONS,
        structured_output_dir,
        details={},
        dataset=DATASET,
        data_split=SPLIT,
        prompt_path=PROMPT_PATH,
        eval_n_limit=EVAL_N_LIMIT,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
    )

    # Always use remote evaluation
    runtime = create_runtime(llm, metadata, args.eval_num_workers)
    
    runtime.run()

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
