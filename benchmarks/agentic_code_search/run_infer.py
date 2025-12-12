import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from benchmarks.utils.dataset import prepare_dataset

# from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
)
from openhands.sdk import LLM, Agent, Conversation, get_logger
from openhands.sdk.conversation import get_agent_final_response
from openhands.sdk.critic import PassCritic
from openhands.sdk.workspace import LocalWorkspace, RemoteWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerWorkspace


logger = get_logger(__name__)


def get_parser():
    """Create and return argument parser.

    Returns:
        ArgumentParser instance
    """
    prompt_dir = (Path(__file__).parent / "prompts").resolve()
    default_prompt_path = prompt_dir / "file_module.j2"
    assert default_prompt_path.exists(), (
        f"Default prompt path file {default_prompt_path} not found"
    )
    parser = ArgumentParser(description="Run inference on code localization dataset")
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Path of the prepared dataset JSONL file.",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--llm-config-path",
        type=str,
        required=True,
        help="Path to JSON LLM configuration",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=25,
        help="Maximum steps allowed for the agent",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of evaluation workers"
    )
    # parser.add_argument("--note", type=str, default="initial", help="Evaluation note")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_outputs",
        help="Evaluation output directory",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=-1,
        help="Limit number of instances to evaluate",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=str(default_prompt_path),
        help="Path to prompt template file",
    )

    parser.add_argument(
        "--runtime",
        type=str,
        default="local",
        choices=["local", "docker"],
        help="Runtime environment for the agent (local or docker)",
    )
    parser.add_argument(
        "--select",
        type=str,
        default="",
        help="Path to text file containing instance IDs to select (one per line)",
    )
    parser.add_argument(
        "--workspace_base_dir",
        type=str,
        default="/tmp/workspace/",
        help="Base directory for local workspaces (ignored for remote workspaces)",
    )
    return parser


def get_instruction(instance: EvalInstance, metadata: EvalMetadata) -> str:
    working_dir = instance.data["repo_dir"]
    problem_statement = instance.data["problem_statement"]
    if metadata.prompt_path is not None:
        prompts_dir = os.path.dirname(metadata.prompt_path)
        template_name = os.path.basename(metadata.prompt_path)
        env = Environment(loader=FileSystemLoader(prompts_dir))
        template = env.get_template(template_name)
        context = {"problem_statement": problem_statement, "working_dir": working_dir}
        instruction = template.render(context)
        return instruction
    else:
        raise ValueError("metadata.prompt_path is None")


def f1_reward_function(predicted_set, true_set):
    if len(true_set) == 0:
        return 0
    tp = len(predicted_set & true_set)
    precision = tp / len(predicted_set) if predicted_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
    if not predicted_set and not true_set:
        return 1.0
    return (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )


def parse_simple_output(raw_output: str, repo_dir: str) -> List:
    # Remove triple backticks and whitespace
    raw_output = raw_output.strip("` \n")
    if not repo_dir.endswith("/"):
        repo_dir += "/"
    locations = []
    current_file = None
    current_class = None
    lines = raw_output.strip().split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            current_file = None
            current_class = None
            continue

        # Check if this is a Python file path
        if line.endswith(".py"):
            current_file = line.strip()
            if current_file.startswith("./"):
                current_file = current_file[2:]  # Remove leading ./
            elif current_file.startswith(repo_dir):
                current_file = current_file[
                    len(repo_dir) :
                ]  # make absolute path relative
            continue

        # Parse class declaration
        if line.startswith("class:"):
            class_name = line[len("class:") :].strip()
            class_name = class_name.split()[0]  # Take first word only
            current_class = class_name
            continue

        # Parse function/method declaration
        if line.startswith("function:") or line.startswith("method:"):
            if not current_file:
                logger.info(f"WARNING: Found function/method without a file: {line}")
                continue

            func_text = line.split(":", 1)[1].strip()
            func_name = func_text.split()[0].strip("() ")

            # Check if function includes class prefix (e.g., "MyClass.my_method")
            if "." in func_name:
                parts = func_name.split(".", 1)
                class_name = parts[0].strip()
                method_name = parts[1].strip()

                locations.append(
                    {"file": current_file, "class": class_name, "function": method_name}
                )
            else:
                # Standalone function or method within current class context
                locations.append(
                    {
                        "file": current_file,
                        "class": current_class,
                        "function": func_name,
                    }
                )
            current_file = None  # Reset current file after function processed
            current_class = None  # Reset current class after function processed

    return locations


def convert_to_entity_format(locations: List) -> List[str]:
    entities = []

    for loc in locations:
        file_path = loc["file"]
        class_name = loc.get("class")
        func_name = loc["function"]

        if class_name:
            entity = f"{file_path}:{class_name}.{func_name}"
        else:
            entity = f"{file_path}:{func_name}"

        entities.append(entity)

    return list(set(entities))


def process_raw_output(raw_output: str, repo_dir: str):
    locations = parse_simple_output(raw_output, repo_dir)

    # Extract unique files
    files = list(dict.fromkeys([loc["file"] for loc in locations]))

    # Extract modules (file:class or file if no class)
    entities = convert_to_entity_format(locations)
    modules = []
    for entity in entities:
        # Extract module (class or just file if standalone function)
        if "." in entity.split(":")[-1]:
            # Has a class - extract it: "file.py:Class.method" → "file.py:Class"
            module = entity.rsplit(".", 1)[0]
        else:
            # No class - use full entity: "file.py:function" → "file.py:function"
            module = entity
        if module not in modules:
            modules.append(module)

    all_found_files = set(files)
    all_found_modules = set(modules)
    all_found_entities = set(entities)
    return all_found_files, all_found_modules, all_found_entities


def reward_function(final_message: str, instance: dict) -> dict:
    try:
        gt_files = []
        gt_modules = []
        gt_entities = []

        for change in instance.get("file_changes", []):
            if "file" in change:
                gt_files.append(change["file"])
            if "changes" in change:
                for module in change["changes"].get("edited_modules", []):
                    gt_modules.append(module)
                for entity in change["changes"].get("edited_entities", []):
                    gt_entities.append(entity)
        gt_files = set(gt_files)
        gt_modules = set(gt_modules)
        gt_entities = set(gt_entities)
    except Exception as e:
        print(f"Error extracting ground truth: {e}")
        return {
            "file_reward": 0,
            "module_reward": 0,
            "entity_reward": 0,
            "prediction": {},
            "ground_truth": {},
        }

    try:
        predicted_files, predicted_modules, predicted_entities = process_raw_output(
            final_message, instance["repo_dir"]
        )
    except Exception as e:
        print(f"Error processing raw output: {e}")
        return {
            "file_reward": 0,
            "module_reward": 0,
            "entity_reward": 0,
            "prediction": {},
            "ground_truth": {
                "files": list(gt_files),
                "modules": list(gt_modules),
                "entities": list(gt_entities),
            },
        }
    try:
        file_f1_score = f1_reward_function(predicted_files, gt_files)
        module_f1_score = f1_reward_function(predicted_modules, gt_modules)
        entity_f1_score = f1_reward_function(predicted_entities, gt_entities)
        return {
            "file_reward": file_f1_score,
            "module_reward": module_f1_score,
            "entity_reward": entity_f1_score,
            "prediction": {
                "files": list(predicted_files),
                "modules": list(predicted_modules),
                "entities": list(predicted_entities),
            },
            "ground_truth": {
                "files": list(gt_files),
                "modules": list(gt_modules),
                "entities": list(gt_entities),
            },
        }
    except Exception as e:
        print(f"Error computing F1 scores: {e}")
        return {
            "file_reward": 0,
            "module_reward": 0,
            "entity_reward": 0,
            "prediction": {
                "files": list(predicted_files),
                "modules": list(predicted_modules),
                "entities": list(predicted_entities),
            },
            "ground_truth": {
                "files": list(gt_files),
                "modules": list(gt_modules),
                "entities": list(gt_entities),
            },
        }


class AgenticCodeSearchEvaluation(Evaluation):
    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up agentic code search evaluation.")

        # Load dataset
        instance_data = []
        with open(self.metadata.dataset, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                instance_data.append(data)
        # convert list to pandas dataframe
        dataset = pd.DataFrame(instance_data)
        dataset = prepare_dataset(
            dataset, self.metadata.eval_limit, self.metadata.selected_instances_file
        )

        instances: List[EvalInstance] = []
        for _, row in dataset.iterrows():
            inst_id = str(row["instance_id"])
            instances.append(EvalInstance(id=inst_id, data=row.to_dict()))

        logger.info("Total instances to process: %d", len(instances))
        return instances

    def prepare_workspace(self, instance: EvalInstance):
        runtime_type = (
            self.metadata.details.get("runtime", "local")
            if isinstance(self.metadata.details, dict)
            else "local"
        )
        repo_name = instance.data["repo"]
        if runtime_type == "local":
            assert isinstance(self.metadata.details, dict)
            working_dir = Path(self.metadata.details["workspace_base_dir"]).resolve()
            # instance_dir_name = f"{repo_name.replace('/', '_')}_{instance_id}"
            # working_dir = output_dir / instance_dir_name
            workspace = LocalWorkspace(working_dir=str(working_dir))
            repo_dir = working_dir / instance.id
            repo_dir = str(repo_dir)
        elif runtime_type == "docker":
            # TODO: directly use prebuilt agent-server image?
            workspace = DockerWorkspace(
                # base_image="nikolaik/python-nodejs:python3.12-nodejs22",
                working_dir="/workspace",
                server_image="ghcr.io/openhands/agent-server:latest-python",
            )
            repo_dir = f"/workspace/{repo_name.split('/')[-1]}/"
        else:
            raise NotImplementedError(f"Unsupported runtime type: {runtime_type}")

        base_commit_id = instance.data["base_commit"]
        instance.data["repo_dir"] = (
            repo_dir  # pass repo_dir to instance data for later use
        )

        # run environment setup commands for cloning repo
        # remove working dir if it already exists
        _ = workspace.execute_command(f"rm -rf {repo_dir}")
        # assert rm_dir.exit_code == 0, (
        #     f"Failed to remove existing working dir: {rm_dir.stderr}"
        # )

        # make new empty directory
        mk_dir = workspace.execute_command(f"mkdir -p {repo_dir}")
        assert mk_dir.exit_code == 0, (
            f"Failed to create repo directory: {mk_dir.stderr}"
        )

        # clone repo inside repo_dir
        repo_url = f"https://github.com/{repo_name}.git"
        clone_repo = workspace.execute_command(
            f"git clone {repo_url} {repo_dir}", timeout=10 * 60
        )
        assert clone_repo.exit_code == 0, f"Failed to clone repo: {clone_repo.stderr}"

        # checkout to base commit
        checkout_commit = workspace.execute_command(
            f"git -C {repo_dir} checkout {base_commit_id}", timeout=5 * 60
        )
        assert checkout_commit.exit_code == 0, (
            f"Failed to checkout to commit {base_commit_id}: {checkout_commit.stderr}"
        )

        logger.info(f"Prepared workspace successfully for instance {instance.id}")
        return workspace

    def evaluate_instance(self, instance, workspace):
        """
        Steps:
        1. Prepare the prompt using Jinja2 template
        2. Create agent with the prompt
        3. Run the agent in a conversation until max iterations or task completion
        4. Collect and return the output
        """
        eval_start_time = time.time()
        instruction = get_instruction(instance, self.metadata)
        # NOTE: the default condenser is LLM-based summarizer in get_default_agent, disabling it for now as done in SWE-Bench. This is why we make agent manually here.
        tools = get_default_tools(enable_browser=False)
        agent = Agent(
            llm=self.metadata.llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
        )

        def _log_event(ev):  # keep it simple
            logger.debug("Event: %s", ev)

        assert isinstance(workspace, RemoteWorkspace) or isinstance(
            workspace, LocalWorkspace
        )
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[_log_event],
            max_iteration_per_run=self.metadata.max_iterations,
        )
        conversation.send_message(instruction)
        conversation.run()
        history = list(map(lambda event: event.model_dump(), conversation.state.events))
        finish_message = get_agent_final_response(conversation.state.events)
        if finish_message == "":
            logger.info("No final response from agent.")
        reward_dict = reward_function(finish_message, instance.data)
        if (
            self.metadata.details is not None
            and self.metadata.details["runtime"] == "local"
        ):
            # clean up workspace after use
            workspace.execute_command(f"rm -rf {instance.data['repo_dir']}")
        eval_time_elapsed = time.time() - eval_start_time
        num_steps = 0
        num_tool_calls = 0
        llm_response_id_set = set()
        for event in history:
            event_src = event.get("source", "")
            llm_response_id = event.get("llm_response_id", "")
            event_kind = event.get("kind", "")
            if not event_src == "agent" or llm_response_id == "":
                continue
            if event_kind == "ActionEvent":
                num_tool_calls += 1
            if event_src == "agent" and llm_response_id != "":
                llm_response_id_set.add(llm_response_id)

        num_steps = len(llm_response_id_set)
        event_list = [event for event in conversation.state.events]

        out = EvalOutput(
            instance_id=instance.id,
            test_result={
                "reward": reward_dict,
                "raw_prediction": finish_message,
                "wall_time_seconds": eval_time_elapsed,
                "num_steps": num_steps,
                "num_tool_calls": num_tool_calls,
            },
            instruction=instruction,
            error=None,
            history=event_list,
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )
        return out


def main():
    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # load LLM configuration
    llm_config_path = args.llm_config_path
    if not os.path.isfile(llm_config_path):
        raise ValueError(f"LLM config file {llm_config_path} does not exist")
    with open(llm_config_path, "r") as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=f"agentic_code_search_{args.dataset_file.split('/')[-1].split('.jsonl')[0]}",
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note="",
    )

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset_file,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={
            "runtime": args.runtime,
            "workspace_base_dir": args.workspace_base_dir,
        },
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=[],
        # max_attempts=args.max_attempts,
        # critic_name=args.critic,
        selected_instances_file=args.select if args.select else None,
        critic=PassCritic(),
        # max_retries=args.max_retries,
    )

    evaluator = AgenticCodeSearchEvaluation(
        metadata=metadata, num_workers=args.num_workers
    )
    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))
    end_time = time.time()
    elapsed_time = end_time - start_time
    # save to a .txt file
    with open(os.path.join(structured_output_dir, "time_taken.txt"), "w") as f:
        f.write(
            f"Time taken for evaluation: {elapsed_time:.2f} seconds, {elapsed_time / 60:.2f} minutes, {elapsed_time / 3600:.2f} hours"
        )

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
