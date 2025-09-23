from __future__ import annotations

import json
import os
import tempfile

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.git_tools import (
    get_git_patch,
    initialize_workspace,
    setup_workspace,
)
from benchmarks.utils.shared import EvalMetadata, EvalOutput
from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Message,
    TextContent,
    get_logger,
)
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


def process_instance_simplified(
    instance: pd.Series, instruction: str, metadata: EvalMetadata
) -> EvalOutput:
    """Process a single instance using the simplified SDK approach."""
    logger.info(f"Starting evaluation for instance {instance.instance_id}")

    temp_workspace = tempfile.mkdtemp()
    workspace_path = setup_workspace(
        instance.repo, instance.base_commit, temp_workspace
    )
    initialize_workspace(
        workspace_path, instance.instance_id, metadata.env_setup_commands
    )

    llm = metadata.llm

    # Setup tools with the workspace
    # tools = [
    #     BashTool.create(working_dir=workspace_path),
    #     FileEditorTool.create(workspace_root=workspace_path),
    # ]

    # Create agent
    agent = Agent(llm=llm)  # tools=tools

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


def run_evaluation(metadata: EvalMetadata):
    """Run evaluation on instances."""
    output_file = os.path.join(metadata.eval_output_dir, "output.jsonl")
    # Load and prepare dataset
    assert metadata.dataset is not None, "Dataset name is required but not provided"
    assert metadata.data_split is not None, "Data split is required but not provided"
    assert metadata.eval_n_limit is not None, "Eval n limit is required but not provided"
    instances = get_dataset(
        metadata.dataset, metadata.data_split, output_file, metadata.eval_n_limit
    )
    print(f"### OUTPUT FILE: {output_file} ###")

    output_dir = os.path.dirname(output_file)
    if output_dir:  # Only create directory if dirname is not empty
        os.makedirs(output_dir, exist_ok=True)

    # Create empty output file
    with open(output_file, "w") as f:
        pass

    results = []
    for idx, instance in instances.iterrows():
        logger.info(f"Processing instance {instance.instance_id}")
        # Get instruction
        workspace_path = os.path.join("/workspace", instance.repo.split("/")[-1])
        assert metadata.prompt_path is not None, "Prompt path is required but not provided"
        instruction = get_instruction(
            instance, metadata, workspace_path, metadata.prompt_path
        )
        result = process_instance_simplified(instance, instruction, metadata)

        # Save result using the complete format
        result_dict = result.model_dump(mode="json")
        if result.error:
            result_dict["error"] = result.error
        results.append(result_dict)

        logger.info(f"Writing result for {instance.instance_id} to {output_file}")
        logger.info(f"Result dict keys: {list(result_dict.keys())}")
        git_patch_len = len(result_dict.get("test_result", {}).get("git_patch", ""))
        logger.info(f"Git patch length: {git_patch_len}")

        # Write to output file
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "a") as f:
            json_line = json.dumps(result_dict) + "\n"
            f.write(json_line)
            f.flush()  # Ensure it's written immediately
            logger.info(
                f"Successfully wrote {len(json_line)} characters to output file"
            )
