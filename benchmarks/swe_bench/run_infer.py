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
    process_instance_simplified,
    read_completed_instances,
)
from benchmarks.utils.runtime import Runtime
from openhands.sdk import Conversation, get_logger
from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation


logger = get_logger(__name__)


def create_runtime(llm: Any, metadata: EvalMetadata, num_workers: int = 1) -> None:
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

            write_output_to_file(result, output_file)

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
