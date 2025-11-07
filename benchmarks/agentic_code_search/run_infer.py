import ast
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import pandas as pd
from jinja2 import Environment, FileSystemLoader

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
    default_prompt_path = prompt_dir / "default.j2"
    assert default_prompt_path.exists(), (
        f"Default prompt path file {default_prompt_path} not found"
    )
    parser = ArgumentParser(description="Run inference on code localization dataset")
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Path of the prepared dataset parquet file.",
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
    # parser.add_argument(
    #     "--num_runs",
    #     type=int,
    #     default=1,
    #     help="Number of evaluation runs for each task instance (useful for best@K evaluation)"
    # )
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


def f1_reward_function(predicted_files, true_files):
    pred, true = set(predicted_files), set(true_files)
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    if not pred and not true:
        return 1.0
    return (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )


def reward_function(
    final_message: str, instance: dict
) -> tuple[float, set[str], set[str]]:
    true_files = set(x[0] for x in ast.literal_eval(instance["target"]))
    score = 0.0
    try:
        predicted_files = set(
            ast.literal_eval(
                final_message.split("<file-list>")[1].split("</file-list>")[0]
            )
        )
        score = f1_reward_function(predicted_files, true_files)
    except Exception as e:
        print(f"Error parsing final message: {e}")
        return 0.0, set(), true_files

    return score, predicted_files, true_files


class AgenticCodeSearchEvaluation(Evaluation):
    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up agentic code search evaluation.")

        # Load dataset
        dataset = pd.read_parquet(self.metadata.dataset)

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
                base_image="nikolaik/python-nodejs:python3.12-nodejs22",
                working_dir="/workspace",
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
        rm_dir = workspace.execute_command(f"rm -rf {repo_dir}")
        assert rm_dir.exit_code == 0, (
            f"Failed to remove existing working dir: {rm_dir.stderr}"
        )

        # make new empty directory
        mk_dir = workspace.execute_command(f"mkdir -p {repo_dir}")
        assert mk_dir.exit_code == 0, (
            f"Failed to create repo directory: {mk_dir.stderr}"
        )

        # clone repo inside repo_dir
        repo_url = f"https://github.com/{repo_name}.git"
        clone_repo = workspace.execute_command(f"git clone {repo_url} {repo_dir}")
        assert clone_repo.exit_code == 0, f"Failed to clone repo: {clone_repo.stderr}"

        # checkout to base commit
        checkout_commit = workspace.execute_command(
            f"git -C {repo_dir} checkout {base_commit_id}"
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
        reward, predicted_files, true_files = reward_function(
            finish_message, instance.data
        )
        out = EvalOutput(
            instance_id=instance.id,
            test_result={
                "reward": reward,
                "predicted_files": list(predicted_files),
                "true_files": list(true_files),
            },
            instruction=instruction,
            error=None,
            history=history,
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )
        return out


def main():
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

    # Load LLM configuration
    with open(args.llm_config_path, "r") as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=f"agentic_code_search_{args.dataset_file.split('.parquet')[0]}",
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
        # max_retries=args.max_retries,
    )

    evaluator = AgenticCodeSearchEvaluation(
        metadata=metadata, num_workers=args.num_workers
    )
    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
