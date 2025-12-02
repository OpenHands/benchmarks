"""Workflow memory utilities for SWE-bench evaluation."""

from pathlib import Path
from typing import Literal

from openhands.sdk.context import Skill
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

WorkflowMode = Literal["none", "offline_no_retrieve", "offline_retrieve", "online"]


def load_workflow_memory(
    mode: WorkflowMode,
    workflow_path: str | None = None,
    instance_data: dict | None = None,
    top_k: int = 3,
) -> Skill | None:
    """Load workflow memory as a Skill.

    Args:
        mode: Workflow memory mode
        workflow_path: Path to workflow file/directory
        instance_data: Instance data for retrieval-based modes
        top_k: Number of workflows to retrieve (for offline_retrieve mode)

    Returns:
        Skill object with workflow memory content, or None if mode is "none"
    """
    if mode == "none":
        return None

    if workflow_path is None:
        # Default to CAWM/workflow/offline_workflow.txt
        default_path = (
            Path(__file__).parent.parent.parent
            / "CAWM"
            / "workflow"
            / "offline_workflow.txt"
        )
        workflow_path = str(default_path)

    workflow_path_obj = Path(workflow_path)

    if mode == "offline_no_retrieve":
        # Load entire workflow file as static memory
        content = _load_offline_workflow(workflow_path_obj)

    elif mode == "offline_retrieve":
        # Retrieve relevant workflows based on instance
        content = _retrieve_workflows(workflow_path_obj, instance_data, top_k)

    elif mode == "online":
        # Future: online workflow generation
        raise NotImplementedError("Online workflow memory mode not yet implemented")

    else:
        raise ValueError(f"Unknown workflow memory mode: {mode}")

    # Return as a Skill with trigger=None (always active)
    return Skill(
        name="workflow_memory",
        content=content,
        source=workflow_path,
        trigger=None,  # Always active = injected into system prompt
    )


def _load_offline_workflow(workflow_path: Path) -> str:
    """Load offline workflow file as-is."""
    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")

    with open(workflow_path) as f:
        content = f.read()

    # Wrap content with instructions
    wrapped_content = f"""# Workflow Memory

You have access to the following workflow patterns that have been successful in solving similar software engineering tasks. Use these as references when approaching the current issue:

{content}

Apply these workflows appropriately based on the current task context. These are guidelines, not strict requirements - adapt them as needed for the specific issue at hand.
"""

    logger.info(f"Loaded offline workflow memory from {workflow_path}")
    return wrapped_content


def _retrieve_workflows(
    workflow_path: Path,
    instance_data: dict | None,
    top_k: int,
) -> str:
    """Retrieve relevant workflows using RAG (future implementation).

    For now, falls back to loading entire workflow file.
    In the future, this will use OpenHands SDK's retrieval capabilities.
    """
    # TODO: Implement retrieval using:
    # - openhands.sdk for embedding/retrieval
    # - Parse workflow file into individual workflows
    # - Embed workflows and instance problem statement
    # - Retrieve top_k most relevant workflows

    logger.warning(
        "Workflow retrieval not yet implemented. "
        "Falling back to offline_no_retrieve mode."
    )
    return _load_offline_workflow(workflow_path)
