"""
Instance class for representing evaluation instances.
"""

from typing import Any, Dict

from pydantic import BaseModel


class Instance(BaseModel):
    """
    Represents a single evaluation instance.
    
    This class provides a structured way to represent instances across different
    benchmarks while maintaining flexibility through the generic data field.
    """
    
    id: str  # Mandatory unique identifier
    data: Dict[str, Any]  # Generic data field for benchmark-specific content