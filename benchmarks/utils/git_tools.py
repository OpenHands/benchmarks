"""Git-related utility functions for SWE-bench evaluation."""

import os
import shutil
import subprocess

from benchmarks.utils.binary_patch_utils import remove_binary_diffs
from benchmarks.utils.shared import EvalException
from openhands.sdk import get_logger


logger = get_logger(__name__)


def setup_workspace(repo_name: str, base_commit: str, workspace_root: str) -> str:
    """Setup workspace by cloning the repository.

    Args:
        repo_name: Repository name (e.g., "django/django")
        base_commit: Git commit hash to checkout
        workspace_root: Root directory for workspace

    Returns:
        Path to the workspace directory
    """
    # Extract directory name from repo name (e.g., "django/django" -> "django")
    workspace_dir_name = repo_name.split("/")[-1]
    workspace_path = os.path.join(workspace_root, workspace_dir_name)

    # Construct GitHub URL
    repo_url = f"https://github.com/{repo_name}.git"

    logger.info(f"Setting up workspace for {repo_name} at {workspace_path}")

    # Remove existing directory if it exists
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path)

    # Clone the repository
    try:
        subprocess.run(
            ["git", "clone", repo_url, workspace_path],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Successfully cloned {repo_url} to {workspace_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository {repo_url}: {e.stderr}")
        raise EvalException(f"Failed to clone repository: {e.stderr}")

    # Checkout the base commit
    try:
        subprocess.run(
            ["git", "checkout", base_commit],
            cwd=workspace_path,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Successfully checked out base commit {base_commit}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to checkout base commit {base_commit}: {e.stderr}")
        raise EvalException(f"Failed to checkout base commit: {e.stderr}")

    return workspace_path


def get_git_patch(workspace_path: str) -> str:
    """Get git patch from the workspace."""
    logger.info("-" * 30)
    logger.info("BEGIN Git Patch Extraction")
    logger.info("-" * 30)

    try:
        # Change to workspace directory
        os.chdir(workspace_path)

        # Configure git
        subprocess.run(
            ["git", "config", "--global", "core.pager", '""'],
            check=True,
            capture_output=True,
            text=True,
        )

        # Remove any nested git repositories
        result = subprocess.run(
            ["find", ".", "-type", "d", "-name", ".git", "-not", "-path", "./.git"],
            capture_output=True,
            text=True,
        )
        git_dirs = [p for p in result.stdout.strip().split("\n") if p]
        for git_dir in git_dirs:
            shutil.rmtree(git_dir)
            logger.info(f"Removed nested git directory: {git_dir}")

        # Check if this is a git repository
        try:
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            logger.error("Current directory is not a git repository")
            return ""

        # Add all changes
        subprocess.run(["git", "add", "-A"], check=True, capture_output=True, text=True)

        # Get the diff
        result = subprocess.run(
            ["git", "diff", "--cached"], capture_output=True, text=True
        )
        git_patch = result.stdout

        # Remove binary diffs if present
        git_patch = remove_binary_diffs(git_patch)

        logger.info(f"Generated git patch with {len(git_patch)} characters")
        return git_patch

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate git patch: {e.stderr}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error generating git patch: {str(e)}")
        return ""


def initialize_workspace(
    workspace_path: str,
    instance_id: str,
    custom_env_setup_commands: list[str] | None = None,
):
    """Initialize the workspace with necessary setup."""
    logger.info("-" * 30)
    logger.info("BEGIN Workspace Initialization")
    logger.info("-" * 30)

    # Set up environment variables and git configuration
    env_setup_commands = [
        f"export SWE_INSTANCE_ID={instance_id}",
        'git config --global core.pager ""',
        "git config --global diff.binary false",
    ]

    # Append custom environment setup commands if provided
    if custom_env_setup_commands:
        env_setup_commands.extend(custom_env_setup_commands)

    for cmd in env_setup_commands:
        try:
            subprocess.run(
                cmd,
                shell=True,
                cwd=workspace_path,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Successfully executed: {cmd}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to execute {cmd}: {e.stderr}")
            raise EvalException(f"Failed to initialize workspace: {e.stderr}")
