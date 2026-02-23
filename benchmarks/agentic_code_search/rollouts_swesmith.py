import os
import shutil
import subprocess
import time
import uuid
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader

from benchmarks.agentic_code_search.custom_agent import CustomAgent
from benchmarks.agentic_code_search.localization_finish import (
    LocalizationFinishAction,
    LocalizationFinishTool,
)
from benchmarks.utils.dataset import get_dataset

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
from openhands.sdk.agent.utils import prepare_llm_messages
from openhands.sdk.conversation import get_agent_final_response
from openhands.sdk.critic import PassCritic
from openhands.sdk.event import ActionEvent
from openhands.sdk.tool import Tool, register_tool
from openhands.sdk.workspace import LocalWorkspace, RemoteWorkspace
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.glob import GlobTool
from openhands.tools.grep import GrepTool
from openhands.tools.planning_file_editor import PlanningFileEditorTool
from openhands.tools.preset.default import get_default_tools
from openhands.tools.terminal import TerminalTool


logger = get_logger(__name__)
register_tool(LocalizationFinishTool.name, LocalizationFinishTool)

# NOTE: Finish and Think tools are included by default if enable_sdk_default_tools is True, else you can only include the below tools in your agent
TOOL_MAP = {
    "grep": GrepTool,
    "terminal": TerminalTool,
    "glob": GlobTool,
    "file_editor": FileEditorTool,
    "planning_file_editor": PlanningFileEditorTool,
    "localization_finish": LocalizationFinishTool,
}


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
        "--dataset",
        type=str,
        required=True,
        help="Name of HuggingFace dataset",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--system_prompt_file",
        type=str,
        default="",
        help="System prompt jinja template file (defaults to OpenHands system prompt)",
    )
    parser.add_argument(
        "--user_prompt_file",
        type=str,
        default=str(default_prompt_path),
        help="User prompt jinja template file (defaults to prompts/file_module.j2)",
    )
    # accept list of tools as argument
    parser.add_argument(
        "--tools",
        type=str,
        nargs="*",
        default=[],
        help="List of tool names to enable for the agent (e.g.: grep, terminal, glob)",
    )
    parser.add_argument(
        "--enable_sdk_default_tools",
        default=False,
        action="store_true",
        help="Whether to enable OpenHands SDK default tools -- Think and Finish tools (do not set to true if using custom finish tool)",
    )
    parser.add_argument(
        "--llm-config-path",
        type=str,
        required=True,
        help="Path to JSON LLM configuration",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum steps allowed for the agent.",
    )
    # parser.add_argument(
    #     "--num_steps_in_prompt",
    #     type=int,
    #     default=-1,
    #     help="Max. number of steps shown to agent (defaults to max-iterations if not set). You can use this to allow some cushioning between the max steps allowed via prompt vs actual max iterations.",
    # )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of evaluation workers"
    )
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
        "--runtime",
        type=str,
        default="local",
        choices=["local"],
        help="Runtime environment for the agent (only local runtimes supported for now)",
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
        default="/tmp/testbed/",
        help="Base directory for local workspaces (ignored for remote workspaces)",
    )
    return parser


def get_instruction(instance: EvalInstance, metadata: EvalMetadata) -> str:
    working_dir = instance.data.get("repo_dir", "./")
    problem_statement = instance.data.get("problem_statement", "")
    user_prompt_path = (
        metadata.details.get("user_prompt_file", None)
        if isinstance(metadata.details, dict)
        else None
    )
    if user_prompt_path is None:
        raise ValueError("args.user_prompt_file is None")
    user_prompt_path = os.path.abspath(user_prompt_path)
    prompts_dir = os.path.dirname(user_prompt_path)
    template_name = os.path.basename(user_prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)
    context = {"problem_statement": problem_statement, "working_dir": working_dir}
    instruction = template.render(context)
    return instruction


def f1_reward_function(predicted_set, true_set):
    if len(true_set) == 0:
        return 0
    tp = len(predicted_set & true_set)
    precision = tp / len(predicted_set) if predicted_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
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


def reward_function_str_parse(final_message: str, instance: dict) -> dict:
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


def parse_structured_outputs(
    structured_locations: List[dict],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Process structured location outputs and extract files, modules, and entities.

    Args:
        structured_locations: List of dicts with 'file', 'class_name', 'function_name' keys
        Returns:
            Tuple of (all_found_files, all_found_modules, all_found_entities) where each is a list of strs

    Example structured input format:
        [
            {'file': 'path/to/file1.py', 'class_name': 'MyClass', 'function_name': 'my_method'},
            {'file': 'path/to/file2.py', 'class_name': None, 'function_name': 'standalone_function'},
            {'file': 'path/to/file1.py', 'class_name': None, 'function_name': 'global_function'},
            {'file': 'path/to/file2.py', 'class_name': 'AnotherClass', 'function_name': None},
            {'file': 'path/to/file3.py', 'class_name': None, 'function_name': None}
        ]

    Example output:
        [['path/to/file1.py', 'path/to/file2.py', 'path/to/file3.py'], ['path/to/file1.py:MyClass', 'path/to/file2.py:AnotherClass', 'path/to/file1.py:global_function', 'path/to/file2.py:standalone_function'], ['path/to/file1.py:MyClass.my_method', 'path/to/file2.py:standalone_function', 'path/to/file1.py:global_function', 'path/to/file2.py:AnotherClass']]
    """

    # Strict sanity check: if there are duplicates in the output, return an empty output so that it is penalized with 0 reward?
    all_found_files = []
    all_found_modules = []
    all_found_entities = []

    for location in structured_locations:
        file_path = location.get("file", None)
        class_name = location.get("class_name", None)
        function_name = location.get("function_name", None)

        # NOTE: Ideally the case of file_path being None should raise an error from the agent-sdk but adding here for safety
        if file_path is None or file_path.strip() == "":
            continue

        all_found_files.append(file_path)

        module = None
        if class_name:
            module = f"{file_path}:{class_name}"
        elif function_name:
            module = f"{file_path}:{function_name}"

        if module:
            all_found_modules.append(module)

        entity = None
        if class_name and function_name:
            entity = f"{file_path}:{class_name}.{function_name}"
        elif function_name:
            entity = f"{file_path}:{function_name}"

        if entity:
            all_found_entities.append(entity)

    all_found_files = list(set(all_found_files))
    all_found_modules = list(set(all_found_modules))
    all_found_entities = list(set(all_found_entities))
    return all_found_files, all_found_modules, all_found_entities


def reward_function_tool_parse(
    structured_locations: Optional[List[Dict[str, Any]]], instance: dict
) -> dict:
    try:
        gt_files = []
        gt_modules = []
        gt_entities = []

        for change in instance.get("file_changes", []):
            if "file" in change:
                gt_files.append(change["file"])
            if "changes" in change:
                edited_modules = change["changes"].get("edited_modules", [])
                edited_modules = [] if edited_modules is None else edited_modules
                for module in edited_modules:
                    gt_modules.append(module)

                edited_entities = change["changes"].get("edited_entities", [])
                edited_entities = [] if edited_entities is None else edited_entities
                for entity in edited_entities:
                    gt_entities.append(entity)
        gt_files = set(gt_files)
        gt_modules = set(gt_modules)
        gt_entities = set(gt_entities)

        if structured_locations is not None:
            predicted_files, predicted_modules, predicted_entities = (
                parse_structured_outputs(structured_locations)
            )
            predicted_files, predicted_modules, predicted_entities = (
                set(predicted_files),
                set(predicted_modules),
                set(predicted_entities),
            )
        else:
            predicted_files, predicted_modules, predicted_entities = set(), set(), set()

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
        print(f"Error computing reward: {e}")
        return {
            "file_reward": 0,
            "module_reward": 0,
            "entity_reward": 0,
            "prediction": {},
            "ground_truth": {},
        }


def get_structured_locations(events) -> Optional[List[Dict[str, Any]]]:
    """Extract structured locations from LocalizationFinishAction in events.
    Args:
        events: List of conversation events to search through.
    Returns:
        List of location dicts with 'file', 'class', 'function' keys, or None if not found.
    """
    # Find the last LocalizationFinishAction
    try:
        for event in reversed(events):
            if (
                isinstance(event, ActionEvent)
                and event.source == "agent"
                and isinstance(event.action, LocalizationFinishAction)
            ):
                # Extract structured locations from the action
                locations = []
                for loc in event.action.locations:
                    locations.append(
                        {
                            "file": loc.file,
                            "class_name": loc.class_name,
                            "function_name": loc.function_name,
                        }
                    )
                return locations
    except Exception as _:
        pass
    return None


class AgenticCodeSearchEvaluation(Evaluation):
    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up agentic code search evaluation.")

        # Load dataset
        dataset = get_dataset(
            dataset_name=self.metadata.dataset,
            split=self.metadata.dataset_split,
            eval_limit=self.metadata.eval_limit,
            selected_instances_file=self.metadata.selected_instances_file,
        )

        instances: List[EvalInstance] = []
        for _, row in dataset.iterrows():
            inst_id = str(row["instance_id"])
            instances.append(EvalInstance(id=inst_id, data=row.to_dict()))

        import random

        random.seed(42)
        random.shuffle(instances)

        logger.info("Total instances to process: %d", len(instances))
        return instances

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace | LocalWorkspace:
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
            uuid_str = str(uuid.uuid4())[:8]
            repo_dir = f"{str(working_dir)}/{uuid_str}/{repo_name.replace('/', '_')}_{instance.id}"
            # delete repo_dir if it already exists
            try:
                shutil.rmtree(repo_dir)
            except Exception as _:
                pass
            # create repo_dir
            os.makedirs(repo_dir, exist_ok=True)
            workspace: RemoteWorkspace | LocalWorkspace = LocalWorkspace(
                working_dir=repo_dir
            )
        else:
            raise NotImplementedError(f"Unsupported runtime type: {runtime_type}")

        instance.data["repo_dir"] = (
            repo_dir  # pass repo_dir to instance.data for later use
        )

        # run environment setup commands for cloning repo

        # clone repo inside repo_dir
        repo_url = f"https://github.com/{repo_name}.git"
        clone_repo = workspace.execute_command(
            f"git clone {repo_url} {repo_dir}", timeout=10 * 60
        )
        assert clone_repo.exit_code == 0, f"Failed to clone repo: {clone_repo.stderr}"

        # checkout to base commit
        # checkout_commit = workspace.execute_command(
        #     f"git -C {repo_dir} checkout {base_commit_id}", timeout=5 * 60
        # )
        # assert checkout_commit.exit_code == 0, (
        #     f"Failed to checkout to commit {base_commit_id}: {checkout_commit.stderr}"
        # )

        patch = instance.data.get("patch")
        assert patch is not None

        # apply patch to the repo by passing patch string to git apply
        subprocess.run(
            ["git", "-C", str(repo_dir), "apply"],
            input=patch,
            check=True,
            capture_output=True,
            text=True,
        )

        logger.info(f"Prepared workspace successfully for instance {instance.id}")
        return workspace

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace | LocalWorkspace
    ) -> EvalOutput:
        """
        Steps:
        1. Prepare the user prompt and the system prompt using Jinja2 template
        2. Create agent with the prompt
        3. Run the agent in a conversation until max iterations or task completion
        4. Collect and return the output
        """
        instruction = get_instruction(instance, self.metadata)
        # NOTE: the default condenser is LLM-based summarizer in get_default_agent, disabling it for now as done in SWE-Bench. This is why we make agent manually here.
        tool_names = (
            self.metadata.details.get("tools", [])
            if isinstance(self.metadata.details, dict)
            else []
        )
        if len(tool_names) > 0:
            tools = []
            for tool_name in tool_names:
                if tool_name in TOOL_MAP:
                    tools.append(Tool(name=TOOL_MAP[tool_name].name))
                else:
                    raise ValueError(
                        f"Unsupported tool name: {tool_name}. Options are: {list(TOOL_MAP.keys())}"
                    )
        else:
            tools = get_default_tools(enable_browser=False)
        system_prompt_path = (
            self.metadata.details.get("system_prompt_file", None)
            if isinstance(self.metadata.details, dict)
            else None
        )

        # create agent
        agent_cls = (
            CustomAgent
            if not self.metadata.details.get("enable_sdk_default_tools", False)  # type: ignore
            else Agent
        )
        if system_prompt_path is None or system_prompt_path == "":
            agent = agent_cls(
                llm=self.metadata.llm,
                tools=tools,
            )
        else:
            # get absolute path of system prompt file
            system_prompt_path = os.path.abspath(system_prompt_path)
            assert os.path.isfile(system_prompt_path), (
                f"System prompt file {system_prompt_path} does not exist"
            )
            agent = agent_cls(
                llm=self.metadata.llm,
                tools=tools,
                system_prompt_filename=str(system_prompt_path),
            )

        def _log_event(ev):  # keep it simple
            logger.debug("Event: %s", ev)

        assert isinstance(workspace, LocalWorkspace)
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[_log_event],
            max_iteration_per_run=self.metadata.max_iterations,
        )
        conversation.send_message(instruction)
        try:
            conversation.run()
        except Exception as e:
            logger.error(f"Error during conversation run: {e}")

        if "localization_finish" not in tool_names:
            finish_message = get_agent_final_response(conversation.state.events)
            if finish_message == "":
                logger.info("No final response from agent.")
            reward_dict = reward_function_str_parse(finish_message, instance.data)
        else:
            structured_locations = get_structured_locations(conversation.state.events)
            reward_dict = reward_function_tool_parse(
                structured_locations, instance.data
            )

        if (
            self.metadata.details is not None
            and self.metadata.details["runtime"] == "local"
        ):
            # clean up workspace after use
            workspace.execute_command(f"rm -rf {instance.data['repo_dir']}")

        event_list = [event for event in conversation.state.events]

        sdk_style_messages = prepare_llm_messages(
            conversation.state.events, condenser=agent.condenser, llm=agent.llm
        )
        llm_messages = agent.llm.format_messages_for_llm(sdk_style_messages)  # type:ignore

        tools = list(agent.tools_map.values())
        cc_tools = []
        if tools:
            cc_tools = [
                t.to_openai_tool(
                    add_security_risk_prediction=True,
                )
                for t in tools
            ]

        out = EvalOutput(
            instance_id=instance.id,
            test_result={
                "reward": reward_dict,
                "llm_messages": llm_messages,
                "tools": cc_tools,
            },
            instruction=instruction,
            error=None,
            history=event_list,
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )
        return out


def main():
    # Set litellm to drop unsupported params globally (for proxy compatibility)
    import litellm

    litellm.drop_params = True

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
    # drop_params=True should be set in config to drop unsupported params like temperature
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=f"agentic_code_search_{args.dataset.split('/')[-1]}",
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note="",
    )

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={
            "runtime": args.runtime,
            "workspace_base_dir": args.workspace_base_dir,
            "system_prompt_file": args.system_prompt_file
            if args.system_prompt_file != ""
            else None,
            "user_prompt_file": args.user_prompt_file,
            "tools": args.tools,
            "enable_sdk_default_tools": args.enable_sdk_default_tools,
            # "num_steps_in_prompt": args.num_steps_in_prompt if args.num_steps_in_prompt > 0 else args.max_iterations,
        },
        prompt_path="",
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
