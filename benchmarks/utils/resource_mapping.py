"""Resource factor mapping utilities for benchmark instances.

Different instances may have different resource requirements.
e.g., some instances may require more memory/CPU to run inference.
This module provides utilities to track per-instance resource requirements.
"""

import json
import os
from pathlib import Path

from openhands.sdk import get_logger

logger = get_logger(__name__)

# Global cache for resource mappings
_global_resource_mapping: dict[str, dict[str, float]] = {}


def get_resource_mapping(
    benchmark_name: str, dataset_name: str
) -> dict[str, float] | None:
    """Load resource mapping for a specific dataset.

    Args:
        benchmark_name: Name of the benchmark (e.g., 'swebench', 'commit0')
        dataset_name: Name of the dataset (e.g., 'princeton-nlp/SWE-bench_Lite')

    Returns:
        Dictionary mapping instance IDs to resource factors, or None if no mapping exists.
    """
    # Normalize dataset name to filename format (replace slashes and special chars)
    # e.g., 'princeton-nlp/SWE-bench_Lite' -> 'princeton-nlp__SWE-bench_Lite'
    dataset_filename = dataset_name.replace("/", "__")
    cache_key = f"{benchmark_name}::{dataset_filename}"

    if cache_key not in _global_resource_mapping:
        # Look for resource mapping file in benchmark's resource directory
        # Try to find the benchmark directory
        benchmarks_dir = Path(__file__).parent.parent
        resource_dir = benchmarks_dir / benchmark_name / "resource"
        mapping_file = resource_dir / f"{dataset_filename}.json"

        if not mapping_file.exists():
            logger.debug(
                f"Resource mapping for {benchmark_name}/{dataset_name} not found at {mapping_file}"
            )
            _global_resource_mapping[cache_key] = {}
            return None

        try:
            with open(mapping_file, "r") as f:
                _global_resource_mapping[cache_key] = json.load(f)
            logger.info(
                f"Loaded resource mapping for {benchmark_name}/{dataset_name} "
                f"with {len(_global_resource_mapping[cache_key])} entries"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load resource mapping from {mapping_file}: {e}"
            )
            _global_resource_mapping[cache_key] = {}
            return None

    mapping = _global_resource_mapping[cache_key]
    return mapping if mapping else None


def get_instance_resource_factor(
    benchmark_name: str, dataset_name: str, instance_id: str, default_factor: int = 1
) -> int:
    """Get the resource factor for a specific instance.

    Args:
        benchmark_name: Name of the benchmark (e.g., 'swebench', 'commit0')
        dataset_name: Name of the dataset (e.g., 'princeton-nlp/SWE-bench_Lite')
        instance_id: Unique identifier for the instance
        default_factor: Default resource factor if no mapping exists (default: 1)

    Returns:
        Resource factor for the instance (integer >= 1).
    """
    resource_mapping = get_resource_mapping(benchmark_name, dataset_name)
    if resource_mapping is None:
        return default_factor

    # Get the factor from mapping, defaulting to default_factor if not found
    factor = resource_mapping.get(instance_id, default_factor)
    return int(factor)
