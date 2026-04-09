"""Utilities for agent context and skills management."""

from openhands.sdk.context import AgentContext
from openhands.sdk.context.skills.skill import load_public_skills


def create_agent_context() -> AgentContext | None:
    """Load public skills and create agent context.

    Respects EXTENSIONS_REF environment variable when loading skills.
    """
    skills = load_public_skills()
    return AgentContext(skills=skills) if skills else None
