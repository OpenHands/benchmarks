"""
Instance class for representing evaluation instances.
"""

from typing import Any, Dict

from pydantic import BaseModel, Field


class Instance(BaseModel):
    """
    Represents a single evaluation instance.

    This class provides a structured way to represent instances across different
    benchmarks while maintaining flexibility through the generic data field.
    """

    id: str = Field(..., description="Mandatory unique identifier")
    data: Dict[str, Any] = Field(..., description="Generic data field for benchmark-specific content")
