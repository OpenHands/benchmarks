from __future__ import annotations

import json
import os
import tempfile
import threading

import pandas as pd
from jinja2 import Environment, FileSystemLoader


from benchmarks.utils.git_tools import (
    get_git_patch,
    initialize_workspace,
    setup_workspace,
)
from benchmarks.utils.shared import EvalMetadata, EvalOutput
from benchmarks.utils.command_tool import create_command_tool
from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Message,
    TextContent,
    get_logger,
)
from openhands.sdk.tool import ToolSpec, register_tool
from openhands.tools.str_replace_editor import FileEditorTool


# from openhands.tools import (
#     BashTool,
#     FileEditorTool,
# )


logger = get_logger(__name__)


def get_instruction(
    instance: pd.Series, metadata: EvalMetadata, workspace_path: str, prompt_path: str
) -> str:
    """Generate instruction for the agent."""
    workspace_dir_name = instance.repo.split("/")[-1]
    assert metadata.details is not None

    # Set up Jinja2 environment
    prompts_dir = os.path.dirname(prompt_path)
    template_name = os.path.basename(prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        "instance": instance,
        "workspace_dir_name": workspace_dir_name,
        "actual_workspace_path": workspace_path,
        "metadata": metadata,
    }

    context["test_instructions"] = ""

    # Render the instruction
    instruction = template.render(context)
    return instruction


def create_workspace_for_instance(instance: pd.Series, metadata: EvalMetadata) -> str:
    """Create workspace for an instance and return the path."""
    temp_workspace = tempfile.mkdtemp()
    workspace_path = setup_workspace(
        instance.repo, instance.base_commit, temp_workspace
    )
    initialize_workspace(
        workspace_path, instance.instance_id, metadata.env_setup_commands
    )
    return workspace_path


def process_instance_simplified(
    instance: pd.Series, instruction: str, metadata: EvalMetadata, workspace_path: str = None
) -> EvalOutput:
    """Process a single instance using the simplified SDK approach."""
    logger.info(f"Starting evaluation for instance {instance.instance_id}")

    if workspace_path is None:
        workspace_path = create_workspace_for_instance(instance, metadata)
    
    temp_workspace = os.path.dirname(workspace_path)

    # Use the original workspace directly - no need to copy
    agent_repo_path = workspace_path
    logger.info(f"Using workspace directly at {agent_repo_path}")

    # Update instruction to include workspace information
    workspace_info = f"\n\nIMPORTANT: The repository is located at {agent_repo_path}. Use this path when working with files and running commands."
    instruction = instruction + workspace_info

    llm = metadata.llm

    # Setup tools - pass the workspace path to command tool
    register_tool("FileEditorTool", FileEditorTool)
    register_tool("CommandTool", lambda: create_command_tool(workspace_path))
    
    tools = [
        ToolSpec(name="FileEditorTool"),
        ToolSpec(name="CommandTool"),
    ]

    # Create agent
    agent = Agent(llm=llm, tools=tools)

    # Create conversation with callback
    conversation = Conversation(agent=agent)

    # Handle multimodal content if present
    if "image_assets" in instance:
        assets = json.loads(str(instance["image_assets"]))
        assert "problem_statement" in assets, (
            "problem_statement is required in image_assets"
        )
        message = Message(
            role="user",
            content=[
                TextContent(text=instruction),
                # TODO: will fix this in next version of SDK
                # ImageContent(image_urls=image_urls),
            ],
        )
    else:
        message = Message(role="user", content=[TextContent(text=instruction)])

    # Send message and run conversation
    conversation.send_message(message)
    conversation.run()

    history = list(conversation.state.events)

    logger.info(f"Conversation completed with {len(history)} events")

    # No need to copy changes back since we're using the original workspace directly
    logger.info(f"Agent worked directly in workspace {agent_repo_path} - no cleanup needed")

    # Get git patch
    git_patch = get_git_patch(workspace_path)

    logger.info(f"Completed evaluation for instance {instance.instance_id}")
    logger.info(f"Git patch length: {len(git_patch)} characters")

    return EvalOutput(
        instance_id=instance.instance_id,
        test_result={
            "git_patch": git_patch,
        },
        instruction=instruction,
        metadata=EvalMetadata(
            llm=metadata.llm,
            max_iterations=metadata.max_iterations,
            eval_output_dir=metadata.eval_output_dir,
            dataset=metadata.dataset,
            data_split=metadata.data_split,
            details=metadata.details,
        ),
        history=history,
    )


def make_metadata(
    llm: LLM,
    dataset_name,
    max_iterations,
    eval_output_dir,
    details=None,
    dataset=None,
    data_split=None,
    prompt_path=None,
    eval_n_limit=None,
    env_setup_commands=None,
):
    """Create evaluation metadata."""
    return EvalMetadata(
        llm=llm,
        data_split=data_split or dataset_name,
        max_iterations=max_iterations,
        eval_output_dir=eval_output_dir,
        details=details,
        dataset=dataset,
        prompt_path=prompt_path,
        eval_n_limit=eval_n_limit,
        env_setup_commands=env_setup_commands,
    )


def construct_eval_output_dir(base_dir, dataset_name, model, max_iterations, eval_note):
    """Construct the structured evaluation output directory path."""
    # Format: eval_out/<dataset>-<split>/<agent_config>/
    # <llm>_maxiter_<maxiter>_N_<version>-<hint>-<exp_name>-run_<run_number>

    # Create LLM config string
    llm_config_str = f"{model}_maxiter_{max_iterations}"

    # Add version and note information
    version = "v1"  # Default version
    hint_status = "no-hint"  # Default hint status

    if eval_note:
        llm_config_str += f"_N_{version}-{hint_status}-{eval_note}-run_1"
    else:
        llm_config_str += f"_N_{version}-{hint_status}-run_1"

    # Construct full path
    eval_output_dir = os.path.join(base_dir, dataset_name, llm_config_str)
    os.makedirs(eval_output_dir, exist_ok=True)

    return eval_output_dir

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

def write_output_to_file(instance, process_instance, result, output_file):
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

