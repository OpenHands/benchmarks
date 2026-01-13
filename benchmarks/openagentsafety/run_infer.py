"""OpenAgentSafety evaluation using OpenHands SDK and shared evaluation framework."""

import fcntl
import json
import os
import subprocess
import time
from typing import Any, List

import numpy as np
import pandas as pd
import requests
from jinja2 import Environment, FileSystemLoader

from benchmarks.openagentsafety.build_images import build_workspace_image
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.conversation import build_event_persistence_callback
from benchmarks.utils.critics import create_critic
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import construct_eval_output_dir
from benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import LLM, Agent, Conversation, get_logger
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerWorkspace


logger = get_logger(__name__)


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif pd.isna(o):
            return None
        return super().default(o)


def download_file(url: str, dest_path: str, max_retries: int = 3) -> bool:
    """Download a file from URL to destination path with retries."""
    for attempt in range(max_retries):
        try:
            logger.info(
                f"Downloading {url} to {dest_path} (attempt {attempt + 1}/{max_retries})"
            )
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(dest_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Successfully downloaded {dest_path}")
            return True

        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                return False
    return False


def download_files_for_task(workspace, instance_data: dict) -> None:
    """Download files as specified in the dataset."""
    # Download workspace files
    if instance_data.get("has_workspace", False):
        workspace_files = instance_data.get("workspace_files", [])
        if workspace_files:
            logger.info(f"Setting up {len(workspace_files)} workspace files")

            for file_url in workspace_files:
                try:
                    filename = file_url.split("/")[-1]

                    # Extract path structure if present
                    if "/workspace/" in file_url:
                        path_parts = file_url.split("/workspace/")[-1]
                        dest_path = f"/workspace/{path_parts}"
                    else:
                        dest_path = f"/workspace/{filename}"

                    # Create parent directories
                    parent_dir = "/".join(dest_path.split("/")[:-1])
                    workspace.execute_command(f"mkdir -p {parent_dir}", timeout=30)

                    # Download directly in container using curl
                    download_cmd = f"curl -fsSL -o {dest_path} '{file_url}'"
                    result = workspace.execute_command(download_cmd, timeout=120)

                    if result.exit_code == 0:
                        # Verify file was downloaded and has content
                        check = workspace.execute_command(
                            f"ls -lh {dest_path} && head -5 {dest_path}", timeout=10
                        )
                        logger.info(f"Downloaded {dest_path}:\n{check.stdout}")

                        # Make executable if script
                        if dest_path.endswith((".py", ".sh", ".bash")):
                            workspace.execute_command(
                                f"chmod +x {dest_path}", timeout=30
                            )
                    else:
                        logger.error(f"Failed to download {file_url}: {result.stderr}")

                except Exception as e:
                    logger.error(f"Error downloading {file_url}: {e}")

    # Download utils files
    if instance_data.get("has_utils", False):
        utils_files = instance_data.get("utils_files", [])
        if utils_files:
            logger.info(f"Setting up {len(utils_files)} utils files")
            workspace.execute_command("mkdir -p /utils", timeout=30)

            for file_url in utils_files:
                try:
                    filename = file_url.split("/")[-1]

                    if "/utils/" in file_url:
                        path_parts = file_url.split("/utils/")[-1]
                        dest_path = f"/utils/{path_parts}"
                    else:
                        dest_path = f"/utils/{filename}"

                    parent_dir = "/".join(dest_path.split("/")[:-1])
                    workspace.execute_command(f"mkdir -p {parent_dir}", timeout=30)

                    download_cmd = f"curl -fsSL -o {dest_path} '{file_url}'"
                    result = workspace.execute_command(download_cmd, timeout=120)

                    if result.exit_code == 0:
                        check = workspace.execute_command(
                            f"ls -lh {dest_path}", timeout=10
                        )
                        logger.info(f"Downloaded {dest_path}:\n{check.stdout}")

                        if dest_path.endswith((".py", ".sh", ".bash")):
                            workspace.execute_command(
                                f"chmod +x {dest_path}", timeout=30
                            )
                    else:
                        logger.error(f"Failed to download {file_url}: {result.stderr}")

                except Exception as e:
                    logger.error(f"Error downloading {file_url}: {e}")


def cleanup_docker_containers():
    """Clean up lingering Docker containers."""
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "-q",
                "--filter",
                "ancestor=openagentsafety-agent-server:local",
            ],
            capture_output=True,
            text=True,
        )
        container_ids = [c for c in result.stdout.strip().split("\n") if c]
        if container_ids:
            logger.info(f"Cleaning up {len(container_ids)} containers")
            subprocess.run(["docker", "rm", "-f"] + container_ids, capture_output=True)
            time.sleep(2)
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")


def setup_host_mapping(workspace):
    """Add the-agent-company.com host mapping inside the container."""
    try:
        gateway_ip = "172.17.0.1"
        logger.info(f"Adding host mapping: {gateway_ip} the-agent-company.com")
        workspace.execute_command(
            f"echo '{gateway_ip} the-agent-company.com' >> /etc/hosts"
        )
        result = workspace.execute_command("grep the-agent-company /etc/hosts")
        logger.info(f"Verification: {result.stdout}")
    except Exception as e:
        logger.error(f"Host mapping error: {e}")


def write_npc_config(
    workspace,
    npc_profiles: List[dict],
    npc_scenarios: List[dict],
    npc_api_key: str,
    npc_base_url: str,
    default_npc_model: str,
    problem_statement: str,
):
    """Write NPC configuration to workspace for the Docker container to read."""
    config = {
        "npc_api_key": npc_api_key,
        "npc_base_url": npc_base_url,
        "default_npc_model": default_npc_model,
        "npc_profiles": npc_profiles,
        "npc_scenarios": npc_scenarios,
        "problem_statement": problem_statement,
    }

    config_json = json.dumps(config, indent=2, cls=NumpyEncoder)
    bash_command = f"""
mkdir -p /npc
cat > /npc/.npc_config.json << 'EOFNPC'
{config_json}
EOFNPC
chmod 600 /npc/.npc_config.json
"""

    try:
        workspace.execute_command(bash_command, timeout=60)
        logger.info("Wrote NPC config to /npc/.npc_config.json")
    except Exception as e:
        logger.error(f"Failed to write NPC config: {e}")
        raise


def generate_instruction(instance_data: dict, template_path: str | None = None) -> str:
    """Generate instruction for the agent using Jinja template."""
    if template_path is None:
        # Use default template
        template_path = os.path.join(os.path.dirname(__file__), "prompts", "default.j2")

    # Set up Jinja2 environment
    prompts_dir = os.path.dirname(template_path)
    template_name = os.path.basename(template_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Render the instruction
    instruction = template.render(instance=instance_data)
    return instruction


def run_evaluation_in_container(
    workspace,
    evaluator_code: str,
    trajectory: str,
    instance_id: str,
    attempt: int = 1,
) -> dict:
    """Execute evaluator code in the Docker container and return results."""
    logger.info(f"Running evaluation for {instance_id} (attempt {attempt})")

    # Write evaluator code
    evaluator_path = "/workspace/evaluator_temp.py"
    bash_command = f"cat > {evaluator_path} << 'EOFEVAL'\n{evaluator_code}\nEOFEVAL"
    workspace.execute_command(bash_command, timeout=60)

    # Write trajectory
    trajectory_path = "/workspace/trajectory_temp.json"
    trajectory_json = json.dumps(trajectory, cls=NumpyEncoder)
    bash_command = f"cat > {trajectory_path} << 'EOFTRAJ'\n{trajectory_json}\nEOFTRAJ"
    workspace.execute_command(bash_command, timeout=60)

    # Create and run evaluation script
    eval_runner = f"""
import sys
import json

sys.path.insert(0, '/workspace')
import evaluator_temp

with open('{trajectory_path}', 'r') as f:
    trajectory = f.read()

try:
    result = evaluator_temp.grade_checkpoints(trajectory=trajectory)
    output = result.to_dict()
    print(json.dumps(output))
except Exception as e:
    import traceback
    print(json.dumps({{"error": str(e), "traceback": traceback.format_exc()}}))
    sys.exit(1)
"""

    runner_path = "/workspace/eval_runner.py"
    bash_command = f"cat > {runner_path} << 'EOFRUNNER'\n{eval_runner}\nEOFRUNNER"
    workspace.execute_command(bash_command, timeout=60)

    result = workspace.execute_command(
        f"cd /workspace && python {runner_path}", timeout=90
    )
    output_str = result.stdout.strip()

    if not output_str:
        logger.error(f"Empty output from evaluator for {instance_id}")
        return {"error": "Empty output from evaluator"}

    try:
        eval_result = json.loads(output_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse evaluator output: {e}")
        logger.error(f"Output was: {output_str[:500]}")
        return {"error": f"JSON decode error: {e}"}

    logger.info(f"Evaluation completed for {instance_id}")
    return eval_result


class OpenAgentSafetyEvaluation(Evaluation):
    """
    OpenAgentSafety evaluation implemented as a child of the
    abstract Evaluation orchestrator.

    Implements:
      - prepare_instances()
      - prepare_workspace(instance)
      - evaluate_instance(instance, workspace)
    """

    def prepare_instances(self) -> List[EvalInstance]:
        """Load OpenAgentSafety dataset into EvalInstance objects."""
        logger.info("Setting up OpenAgentSafety evaluation data")

        df = get_dataset(
            dataset_name=self.metadata.dataset,
            split=self.metadata.dataset_split,
            eval_limit=self.metadata.eval_limit,
            selected_instances_file=self.metadata.selected_instances_file,
        )

        instances: List[EvalInstance] = []
        for _, row in df.iterrows():
            inst_id = str(row["instance_id"])
            # Convert numpy types to Python types
            data = convert_numpy_types(row.to_dict())
            # Ensure data is a dict
            if not isinstance(data, dict):
                raise ValueError(f"Expected dict, got {type(data)}")
            instances.append(EvalInstance(id=inst_id, data=data))

        logger.info("Total instances to process: %d", len(instances))
        return instances

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """Create a fresh Docker workspace for this instance.

        Args:
            instance: The evaluation instance to prepare workspace for.
            resource_factor: Resource factor for runtime allocation (default: 1).
            forward_env: Environment variables to forward into the workspace.
        """
        server_image = build_workspace_image()

        workspace = DockerWorkspace(
            server_image=server_image,
            platform="linux/amd64",
            extra_ports=True,
            forward_env=forward_env or [],
        )

        # Setup host mapping for The Agent Company services
        setup_host_mapping(workspace)

        download_files_for_task(workspace, instance.data)

        # Setup NPC config if needed
        if instance.data.get("npcs", 0) > 0:
            npc_api_key = os.getenv("NPC_API_KEY", "")
            npc_base_url = os.getenv("NPC_BASE_URL") or self.metadata.llm.base_url or ""
            npc_model = os.getenv("NPC_MODEL", "litellm_proxy/openai/gpt-4o")

            write_npc_config(
                workspace=workspace,
                npc_profiles=instance.data["agent_profiles"],
                npc_scenarios=instance.data["agent_scenarios"],
                npc_api_key=npc_api_key,
                npc_base_url=npc_base_url,
                default_npc_model=npc_model,
                problem_statement=instance.data["problem_statement"],
            )

        return workspace

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """Run the agent on one instance and return evaluation results."""
        import warnings

        from pydantic import ValidationError

        # Setup tools
        tools = get_default_tools(
            enable_browser=False,
        )

        # Create agent
        agent = Agent(llm=self.metadata.llm, tools=tools)

        # Collect events
        received_events = []

        def event_callback(event) -> None:
            """Collect all events, filtering out state updates."""
            from openhands.sdk.event.conversation_state import (
                ConversationStateUpdateEvent,
            )

            if not isinstance(event, ConversationStateUpdateEvent):
                received_events.append(event)

        persist_callback = build_event_persistence_callback(
            run_id=self.metadata.eval_output_dir,
            instance_id=instance.id,
            attempt=self.current_attempt,
        )

        # Create conversation
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[persist_callback, event_callback],
            max_iteration_per_run=self.metadata.max_iterations,
            stuck_detection=True,
        )

        # Generate instruction
        instruction = generate_instruction(instance.data)
        conversation.send_message(instruction)

        # Run conversation with error handling and fake user responses
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                run_conversation_with_fake_user_response(conversation)
            logger.info(f"Conversation completed for {instance.id}")
        except ValidationError as e:
            logger.warning(f"Validation error from custom events (continuing): {e}")
        except Exception as e:
            logger.error(f"Error during conversation: {e}")
            # Collect cost metrics even on error
            metrics = None
            if hasattr(self.metadata.llm, "metrics"):
                metrics = self.metadata.llm.metrics
            return EvalOutput(
                instance_id=instance.id,
                attempt=self.current_attempt,
                test_result={"error": str(e)},
                instruction=instruction,
                error=str(e),
                history=[],
                metrics=metrics,
            )

        # Build history safely
        history = []
        for event in received_events:
            try:
                history.append(event.model_dump())
            except Exception:
                # Fallback for events that can't be serialized
                history.append(
                    {"type": type(event).__name__, "string_repr": str(event)}
                )

        trajectory = "\n".join([str(event) for event in received_events])

        # Run evaluation
        eval_result = {}
        if "evaluator_code" in instance.data and instance.data["evaluator_code"]:
            try:
                eval_result = run_evaluation_in_container(
                    workspace=workspace,
                    evaluator_code=instance.data["evaluator_code"],
                    trajectory=trajectory,
                    instance_id=instance.id,
                    attempt=self.current_attempt,
                )
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                eval_result = {"error": f"Evaluation failed: {e}"}
        else:
            logger.warning(f"No evaluator_code for {instance.id}")
            eval_result = {"error": "No evaluator code provided"}

        # Collect cost metrics from LLM
        metrics = None
        if hasattr(self.metadata.llm, "metrics"):
            metrics = self.metadata.llm.metrics
        return EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result=eval_result,
            instruction=instruction,
            error=None if not eval_result.get("error") else eval_result["error"],
            history=history,
            metadata=self.metadata,
            instance=instance.data,
            metrics=metrics,
        )


def main() -> None:
    """Main entry point."""
    parser = get_parser(add_llm_config=True)
    # OpenAgentSafety-specific arguments here if needed

    args = parser.parse_args()

    # Validate args
    if args.max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {args.max_attempts}")

    # Load LLM config
    llm_config_path = args.llm_config_path
    if not os.path.isfile(llm_config_path):
        raise ValueError(f"LLM config file {llm_config_path} does not exist")
    with open(llm_config_path, "r") as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    # Construct output directory
    dataset_description = (
        args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__")
    )

    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )

    # Create critic instance from parsed arguments
    critic = create_critic(args)

    # Create metadata
    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={
            "server_image": "openagentsafety-agent-server:local",
            "platform": "linux/amd64",
        },
        eval_limit=args.n_limit,
        max_attempts=args.max_attempts,
        critic=critic,
        selected_instances_file=args.select,
    )

    # Initial cleanup
    cleanup_docker_containers()

    # Create evaluator
    evaluator = OpenAgentSafetyEvaluation(
        metadata=metadata, num_workers=args.num_workers
    )

    # Define result writer with file locking
    def _default_on_result_writer(eval_output_dir: str):
        def _cb(instance: EvalInstance, out: EvalOutput) -> None:
            try:
                # Write to JSONL with exclusive lock
                with open(evaluator.output_path, "a") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    output_dict = out.model_dump()
                    # Clean up any remaining numpy types
                    output_dict = convert_numpy_types(output_dict)
                    json_str = json.dumps(output_dict)
                    f.write(json_str + "\n")
                    fcntl.flock(f, fcntl.LOCK_UN)
            except Exception as e:
                logger.warning(f"Failed to write to attempt file: {e}")

            # Save individual files
            output_dir = eval_output_dir
            os.makedirs(output_dir, exist_ok=True)

            # Save trajectory
            traj_file = os.path.join(output_dir, f"traj_{instance.id}.json")
            with open(traj_file, "w") as f:
                json.dump(out.history, f, indent=2, cls=NumpyEncoder)

            # Save eval result
            eval_file = os.path.join(output_dir, f"eval_{instance.id}.json")
            with open(eval_file, "w") as f:
                json.dump(out.test_result, f, indent=2, cls=NumpyEncoder)

            # Save state
            state_file = os.path.join(output_dir, f"state_{instance.id}.json")
            state_data = {
                "instance_id": instance.id,
                "history": out.history,
                "num_events": len(out.history) if out.history else 0,
            }
            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2, cls=NumpyEncoder)

        return _cb

    # Run evaluation
    evaluator.run(on_result=_default_on_result_writer(metadata.eval_output_dir))

    # Final cleanup
    cleanup_docker_containers()

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
