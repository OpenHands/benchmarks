from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import tempfile
import subprocess
import shutil
from typing import Any, Literal

import pandas as pd
import toml
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader
from pydantic import SecretStr

# Ensure OpenHands SDK is importable
_SDK_DIR = os.environ.get('OPENHANDS_SDK')
if not _SDK_DIR:
    raise RuntimeError(
        "OPENHANDS_SDK environment variable is not set. "
        "Please set it to the path of your OpenHands SDK directory. "
        "Example: export OPENHANDS_SDK=/path/to/agent-sdk"
    )
if _SDK_DIR not in sys.path:
    sys.path.insert(0, _SDK_DIR)

# Import SDK components
from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    LLMConvertibleEvent,
    Message,
    TextContent,
    ImageContent,
    Tool,
    get_logger,
)
from openhands.tools import (
    BashTool,
    FileEditorTool,
)

# Import SWE-bench specific components
from swe_bench.binary_patch_utils import (
    remove_binary_diffs,
    remove_binary_files_from_git,
)
from swe_bench.resource.mapping import get_instance_resource_factor
from swe_bench.resource.swt_bench_constants import (
    MAP_REPO_TO_INSTALL,
    MAP_REPO_TO_TEST_FRAMEWORK_VERBOSE,
    MAP_VERSION_TO_INSTALL,
)

logger = get_logger('swe_bench_eval')

USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'
ENABLE_LLM_EDITOR = os.environ.get('ENABLE_LLM_EDITOR', 'false').lower() == 'true'
BenchMode = Literal['swe', 'swt', 'swt-ci']

# Global variable to track dataset type
DATASET_TYPE = 'SWE-bench'

class EvalException(Exception):
    """Exception raised during evaluation."""
    pass

class EvalMetadata:
    """Metadata for evaluation."""
    def __init__(self, llm_config, dataset_name, agent_class, max_iterations, eval_note, eval_output_dir, details=None):
        self.llm_config = llm_config
        self.dataset_name = dataset_name
        self.agent_class = agent_class
        self.max_iterations = max_iterations
        self.eval_note = eval_note
        self.eval_output_dir = eval_output_dir
        self.details = details or {}

class EvalOutput:
    """Output from evaluation."""
    def __init__(self, instance_id, git_patch, error=None):
        self.instance_id = instance_id
        self.git_patch = git_patch
        self.error = error

class LLMConfig:
    """LLM configuration."""
    def __init__(self, model, api_key, base_url=None, temperature=0):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.log_completions = True
        self.modify_params = False

def set_dataset_type(dataset_name: str) -> str:
    """Set dataset type based on dataset name."""
    global DATASET_TYPE
    name_lower = dataset_name.lower()

    if 'swe-gym' in name_lower:
        DATASET_TYPE = 'SWE-Gym'
    elif 'swe-bench-live' in name_lower:
        DATASET_TYPE = 'SWE-bench-Live'
    elif 'multimodal' in name_lower:
        DATASET_TYPE = 'Multimodal'
    else:
        DATASET_TYPE = 'SWE-bench'

    logger.info(f'Dataset type set to: {DATASET_TYPE}')

def _get_workspace_dir_name(instance: pd.Series) -> str:
    """Extract repo name from instance.repo (e.g., "django/django" -> "django")"""
    repo_name = instance.repo.split('/')[-1]
    return repo_name

def setup_workspace(instance: pd.Series, workspace_root: str) -> str:
    """Setup workspace for the instance by cloning the repository."""
    repo_name = instance.repo  # e.g., "django/django"
    base_commit = instance.base_commit
    workspace_dir_name = _get_workspace_dir_name(instance)
    workspace_path = os.path.join(workspace_root, workspace_dir_name)
    
    # Construct GitHub URL
    repo_url = f"https://github.com/{repo_name}.git"
    
    logger.info(f'Setting up workspace for {repo_name} at {workspace_path}')
    
    # Remove existing directory if it exists
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path)
    
    # Clone the repository
    try:
        subprocess.run(['git', 'clone', repo_url, workspace_path], check=True, capture_output=True, text=True)
        logger.info(f'Successfully cloned {repo_url} to {workspace_path}')
    except subprocess.CalledProcessError as e:
        logger.error(f'Failed to clone repository {repo_url}: {e.stderr}')
        raise EvalException(f'Failed to clone repository: {e.stderr}')
    
    # Checkout the base commit
    try:
        subprocess.run(['git', 'checkout', base_commit], cwd=workspace_path, check=True, capture_output=True, text=True)
        logger.info(f'Successfully checked out base commit {base_commit}')
    except subprocess.CalledProcessError as e:
        logger.error(f'Failed to checkout base commit {base_commit}: {e.stderr}')
        raise EvalException(f'Failed to checkout base commit: {e.stderr}')
    
    return workspace_path

def get_instruction(instance: pd.Series, metadata: EvalMetadata, workspace_path: str) -> str:
    """Generate instruction for the agent."""
    workspace_dir_name = _get_workspace_dir_name(instance)
    mode = metadata.details['mode']
    llm_model = metadata.llm_config.model

    # Determine the template file based on mode and LLM
    if mode.startswith('swt'):
        template_name = 'swt.j2'
    elif mode == 'swe':
        if 'claude' in llm_model:
            template_name = 'swe_default.j2'
        elif 'gpt-4.1' in llm_model:
            template_name = 'swe_gpt4.j2'
        else:
            template_name = 'swe_default.j2'  # Default for 'swe' mode
    else:
        logger.error(f'Unexpected evaluation mode: {mode}. Falling back to default.')
        template_name = 'swe_default.j2'

    # Set up Jinja2 environment
    prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        'instance': instance,
        'workspace_dir_name': workspace_dir_name,
        'actual_workspace_path': workspace_path,
        'metadata': metadata,
    }

    # Add specific context for swt-ci mode if needed
    if mode == 'swt-ci':
        context['test_instructions'] = (
            f'The following command can be used to run the tests: `{list(MAP_REPO_TO_TEST_FRAMEWORK_VERBOSE[instance.repo].values())[0]}`. Make sure they fail in the expected way.\n'
        )
    else:
        context['test_instructions'] = ''

    # Render the instruction
    instruction = template.render(context)

    if RUN_WITH_BROWSING:
        instruction += (
            '<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web. </IMPORTANT!>\n'
        )

    return instruction

def initialize_workspace(workspace_path: str, instance: pd.Series):
    """Initialize the workspace with necessary setup."""
    logger.info('-' * 30)
    logger.info('BEGIN Workspace Initialization')
    logger.info('-' * 30)
    
    # Set up environment variables and git configuration
    env_setup_commands = [
        f"export SWE_INSTANCE_ID={instance['instance_id']}",
        "export PIP_CACHE_DIR=~/.cache/pip",
        'git config --global core.pager ""',
        'git config --global diff.binary false',
    ]
    
    for cmd in env_setup_commands:
        try:
            subprocess.run(cmd, shell=True, cwd=workspace_path, check=True, capture_output=True, text=True)
            logger.info(f'Successfully executed: {cmd}')
        except subprocess.CalledProcessError as e:
            logger.error(f'Failed to execute {cmd}: {e.stderr}')
            raise EvalException(f'Failed to initialize workspace: {e.stderr}')
    
    # Create necessary directories
    swe_util_dir = '/swe_util/eval_data/instances'
    os.makedirs(swe_util_dir, exist_ok=True)
    
    # Write instance data
    swe_instance_json_name = 'swe-bench-instance.json'
    instance_file_path = os.path.join(swe_util_dir, swe_instance_json_name)
    with open(instance_file_path, 'w') as f:
        if not isinstance(instance, dict):
            json.dump([instance.to_dict()], f)
        else:
            json.dump([instance], f)
    
    # Copy setup scripts
    script_dir = os.path.dirname(__file__)
    if DATASET_TYPE == 'SWE-bench-Live':
        entry_script_path = 'instance_swe_entry_live.sh'
    else:
        entry_script_path = 'instance_swe_entry.sh'
    
    src_script = os.path.join(script_dir, f'scripts/setup/{entry_script_path}')
    dst_script = f'/swe_util/{entry_script_path}'
    if os.path.exists(src_script):
        shutil.copy2(src_script, dst_script)
        
        # Execute the setup script
        try:
            subprocess.run(f'source {dst_script}', shell=True, cwd=workspace_path, check=True, capture_output=True, text=True)
            logger.info(f'Successfully executed setup script: {entry_script_path}')
        except subprocess.CalledProcessError as e:
            logger.warning(f'Setup script execution failed (non-fatal): {e.stderr}')

def get_git_patch(workspace_path: str) -> str:
    """Get git patch from the workspace."""
    logger.info('-' * 30)
    logger.info('BEGIN Git Patch Extraction')
    logger.info('-' * 30)
    
    try:
        # Change to workspace directory
        os.chdir(workspace_path)
        
        # Configure git
        subprocess.run(['git', 'config', '--global', 'core.pager', '""'], check=True, capture_output=True, text=True)
        
        # Remove any nested git repositories
        result = subprocess.run(['find', '.', '-type', 'd', '-name', '.git', '-not', '-path', './.git'], 
                              capture_output=True, text=True)
        git_dirs = [p for p in result.stdout.strip().split('\n') if p]
        for git_dir in git_dirs:
            shutil.rmtree(git_dir)
            logger.info(f'Removed nested git directory: {git_dir}')
        
        # Check if this is a git repository
        try:
            subprocess.run(['git', 'rev-parse', '--git-dir'], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            logger.error('Current directory is not a git repository')
            return ""
        
        # Add all changes
        subprocess.run(['git', 'add', '-A'], check=True, capture_output=True, text=True)
        
        # Get the diff
        result = subprocess.run(['git', 'diff', '--cached'], capture_output=True, text=True)
        git_patch = result.stdout
        
        # Remove binary diffs if present
        git_patch = remove_binary_diffs(git_patch)
        
        logger.info(f'Generated git patch with {len(git_patch)} characters')
        return git_patch
        
    except subprocess.CalledProcessError as e:
        logger.error(f'Failed to generate git patch: {e.stderr}')
        return ""
    except Exception as e:
        logger.error(f'Unexpected error generating git patch: {str(e)}')
        return ""

def process_instance_simplified(instance: pd.Series, metadata: EvalMetadata) -> EvalOutput:
    """Process a single instance using the simplified SDK approach."""
    logger.info(f'Starting evaluation for instance {instance.instance_id}')
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_workspace:
        try:
            # Setup workspace
            workspace_path = setup_workspace(instance, temp_workspace)
            initialize_workspace(workspace_path, instance)
            
            # Configure LLM
            api_key = os.getenv("LITELLM_API_KEY")
            if not api_key:
                raise EvalException("LITELLM_API_KEY environment variable is not set")
            
            llm = LLM(
                model=metadata.llm_config.model,
                base_url=metadata.llm_config.base_url or "https://llm-proxy.eval.all-hands.dev",
                api_key=SecretStr(api_key),
                temperature=metadata.llm_config.temperature,
            )
            
            # Setup tools with the workspace
            tools: list[Tool] = [
                BashTool(working_dir=workspace_path),
                FileEditorTool(),
            ]
            
            # Create agent
            agent = Agent(llm=llm, tools=tools)
            
            # Create conversation
            conversation = Conversation(agent=agent)
            
            # Get instruction
            instruction = get_instruction(instance, metadata, workspace_path)
            
            # Handle multimodal content if present
            if 'image_assets' in instance:
                assets = json.loads(instance['image_assets'])
                assert 'problem_statement' in assets, 'problem_statement is required in image_assets'
                image_urls = assets['problem_statement']
                message = Message(
                    role="user",
                    content=[
                        TextContent(text=instruction),
                        ImageContent(image_urls=image_urls)
                    ]
                )
            else:
                message = Message(
                    role="user",
                    content=[TextContent(text=instruction)]
                )
            
            # Send message and run conversation
            conversation.send_message(message)
            conversation.run()
            
            # Get git patch
            git_patch = get_git_patch(workspace_path)
            
            logger.info(f'Completed evaluation for instance {instance.instance_id}')
            logger.info(f'Git patch length: {len(git_patch)} characters')
            
            return EvalOutput(instance.instance_id, git_patch)
            
        except Exception as e:
            logger.error(f'Error processing instance {instance.instance_id}: {str(e)}')
            return EvalOutput(instance.instance_id, "", error=str(e))

def get_evaluation_parser():
    """Get argument parser for evaluation."""
    import argparse
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--agent-cls', dest='agent_cls', type=str, default=os.environ.get('AGENT_CLS', 'CodeActAgent'))
    parser.add_argument('--max-iterations', dest='max_iterations', type=int, default=int(os.environ.get('MAX_ITERATIONS', '50')))
    parser.add_argument('--eval-output-dir', dest='eval_output_dir', type=str, default=os.environ.get('EVAL_OUTPUT_DIR', './eval_out'))
    parser.add_argument('--eval-num-workers', dest='eval_num_workers', type=int, default=int(os.environ.get('EVAL_NUM_WORKERS', '1')))
    parser.add_argument('--eval-n-limit', dest='eval_n_limit', type=int, default=int(os.environ.get('EVAL_N_LIMIT', '0')))
    parser.add_argument('--eval-note', dest='eval_note', type=str, default=os.environ.get('EVAL_NOTE', ''))
    parser.add_argument('--model', dest='model', type=str, default=os.environ.get('MODEL', 'claude-3-5-sonnet-latest'))
    parser.add_argument('--llm-config', dest='llm_config', type=str, default=os.environ.get('LLM_CONFIG', None))
    return parser

def filter_dataset(dataset: pd.DataFrame, filter_column: str) -> pd.DataFrame:
    """Filter dataset based on environment variables."""
    # This is a simplified version - you may need to add more filtering logic
    return dataset

def make_metadata(llm_config, dataset_name, agent_class, max_iterations, eval_note, eval_output_dir, details=None):
    """Create evaluation metadata."""
    return EvalMetadata(llm_config, dataset_name, agent_class, max_iterations, eval_note, eval_output_dir, details)

def prepare_dataset(dataset: pd.DataFrame, output_file: str, n_limit: int) -> pd.DataFrame:
    """Prepare dataset for evaluation."""
    if n_limit > 0:
        dataset = dataset.head(n_limit)
    return dataset

def run_evaluation_simplified(instances: pd.DataFrame, metadata: EvalMetadata, output_file: str):
    """Run evaluation on instances."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    results = []
    for idx, instance in instances.iterrows():
        try:
            result = process_instance_simplified(instance, metadata)
            
            # Save result
            result_dict = {
                'instance_id': result.instance_id,
                'git_patch': result.git_patch,
                'error': result.error,
            }
            results.append(result_dict)
            
            # Write to output file
            with open(output_file, 'a') as f:
                f.write(json.dumps(result_dict) + '\n')
                
        except Exception as e:
            logger.error(f'Failed to process instance {instance.instance_id}: {str(e)}')
            error_result = {
                'instance_id': instance.instance_id,
                'git_patch': '',
                'error': str(e),
            }
            results.append(error_result)
            
            with open(output_file, 'a') as f:
                f.write(json.dumps(error_result) + '\n')

if __name__ == '__main__':
    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='princeton-nlp/SWE-bench',
        help='data set to evaluate on, either full-test or lite-test',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='swe',
        choices=['swe', 'swt', 'swt-ci'],
        help="mode to run the evaluation, either 'swe', 'swt', or 'swt-ci'",
    )

    args, _ = parser.parse_known_args()

    # Load dataset
    dataset = load_dataset(args.dataset, split=args.split)
    set_dataset_type(args.dataset)

    swe_bench_tests = filter_dataset(dataset.to_pandas(), 'instance_id')
    logger.info(f'Loaded dataset {args.dataset} with split {args.split}: {len(swe_bench_tests)} tasks')

    # Create LLM config
    llm_config = LLMConfig(
        model=args.model,
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url="https://llm-proxy.eval.all-hands.dev",
        temperature=0
    )

    if not llm_config.api_key:
        raise ValueError("LITELLM_API_KEY environment variable is not set")

    details = {'mode': args.mode}
    dataset_description = args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    
    metadata = make_metadata(
        llm_config,
        dataset_description,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details=details,
    )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    print(f'### OUTPUT FILE: {output_file} ###')

    # Prepare dataset
    instances = prepare_dataset(swe_bench_tests, output_file, args.eval_n_limit)
    
    # Run evaluation
    run_evaluation_simplified(instances, metadata, output_file)
    
    logger.info('Evaluation completed!')