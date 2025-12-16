from __future__ import annotations

from typing import cast

import pandas as pd
from datasets import Dataset, load_dataset

from openhands.sdk import get_logger


logger = get_logger(__name__)


def _load_selected_instances(select_file_path: str) -> set[str]:
    """Load instance IDs from a text file (one per line).

    Args:
        select_file_path: Path to text file containing instance IDs

    Returns:
        Set of instance IDs

    Raises:
        FileNotFoundError: If the select file doesn't exist
        ValueError: If the file is empty
    """
    import os

    if not os.path.isfile(select_file_path):
        raise FileNotFoundError(f"Select file not found: {select_file_path}")

    selected_instances = set()
    with open(select_file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                selected_instances.add(line)

    if not selected_instances:
        raise ValueError(
            f"Select file is empty or contains no valid instance IDs: "
            f"{select_file_path}"
        )

    logger.info(
        f"Loaded {len(selected_instances)} instance IDs from {select_file_path}"
    )
    return selected_instances


def prepare_dataset(
    dataset: pd.DataFrame,
    n_limit: int | None = None,
    selected_instances_file: str | None = None,
    instance_ids: str | None = None,
) -> pd.DataFrame:
    """Prepare dataset for evaluation."""
    filtered = False

    # Filter by instance_ids if provided (takes precedence over selected_instances_file)
    if instance_ids:
        instance_id_list = [
            id.strip() for id in instance_ids.split(",") if id.strip()
        ]
        if instance_id_list:
            original_size = len(dataset)
            mask = dataset["instance_id"].isin(instance_id_list)
            dataset = cast(pd.DataFrame, dataset[mask])
            filtered = True
            logger.info(
                f"Filtered to {len(dataset)} instances from instance_ids parameter "
                f"(original size: {original_size})"
            )

    # Filter to selected instances first (if provided)
    elif selected_instances_file:
        selected_instances = _load_selected_instances(selected_instances_file)
        original_size = len(dataset)
        mask = dataset["instance_id"].isin(list(selected_instances))
        dataset = cast(pd.DataFrame, dataset[mask])
        filtered = True
        logger.info(
            f"Selected {len(dataset)} instances from {original_size} total instances"
        )

    # Apply limit after filtering completed instances
    if n_limit is not None and n_limit > 0 and not filtered:
        dataset = dataset.sample(n=min(n_limit, len(dataset)), random_state=42)
    elif n_limit and filtered:
        logger.info(
            "n_limit provided but skipped because instance filtering was requested"
        )

    return dataset


def get_dataset(
    dataset_name: str,
    split: str,
    eval_limit: int | None = None,
    selected_instances_file: str | None = None,
    instance_ids: str | None = None,
) -> pd.DataFrame:
    """Load and prepare dataset for evaluation."""
    # Load dataset
    dataset = load_dataset(dataset_name, split=split)
    assert isinstance(dataset, Dataset)
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # TODO: Add the ability to filter dataset
    logger.info(f"Loaded dataset {dataset_name} with split {split}: {len(df)} tasks")

    # Prepare dataset (apply n_limit if specified and filter selected)
    instances = prepare_dataset(df, eval_limit, selected_instances_file, instance_ids)
    return instances
