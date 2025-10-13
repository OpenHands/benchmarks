from __future__ import annotations

from typing import Optional

import pandas as pd
from datasets import Dataset, load_dataset

from benchmarks.utils.models import EvalInstanceID
from openhands.sdk import get_logger


logger = get_logger(__name__)


def prepare_dataset(
    dataset: pd.DataFrame,
    n_limit: int | None = None,
    completed_instances: Optional[set] = None,
) -> pd.DataFrame:
    """Prepare dataset for evaluation."""

    # Filter out completed instances
    if completed_instances:
        original_size = len(dataset)
        # pandas boolean should return a DataFrame
        dataset = dataset[~dataset["instance_id"].isin(completed_instances)]  # type: ignore
        logger.info(f"Filtered out {original_size - len(dataset)} completed instances")
        logger.info(f"{len(dataset)} instances remaining")

    # Apply limit after filtering completed instances
    if n_limit is not None and n_limit > 0:
        dataset = dataset.head(n_limit)

    return dataset


def get_dataset(
    dataset_name: str,
    split: str,
    eval_limit: int | None = None,
    completed_instances: set[EvalInstanceID] | None = None,
) -> pd.DataFrame:
    """Load and prepare dataset for evaluation."""
    # Load dataset
    dataset = load_dataset(dataset_name, split=split)
    assert isinstance(dataset, Dataset)
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # TODO: Add the ability to filter dataset
    logger.info(f"Loaded dataset {dataset_name} with split {split}: {len(df)} tasks")

    # Prepare dataset (apply n_limit if specified and filter completed)
    instances = prepare_dataset(df, eval_limit, completed_instances)
    return instances
