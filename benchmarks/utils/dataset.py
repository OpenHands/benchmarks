"""Dataset utilities for SWE-bench evaluation."""

from __future__ import annotations

import pandas as pd
from datasets import Dataset, load_dataset

from openhands.sdk import get_logger


logger = get_logger(__name__)


def filter_dataset(dataset: pd.DataFrame, filter_column: str) -> pd.DataFrame:
    """Filter dataset based on environment variables."""
    # This is a simplified version - you may need to add more filtering logic
    return dataset


def prepare_dataset(
    dataset: pd.DataFrame, output_file: str, n_limit: int
) -> pd.DataFrame:
    """Prepare dataset for evaluation."""
    if n_limit > 0:
        dataset = dataset.head(n_limit)
    return dataset


def get_dataset(
    dataset_name: str, split: str, output_file: str, eval_n_limit: int
) -> pd.DataFrame:
    """Load and prepare dataset for evaluation."""
    # Load dataset
    dataset = load_dataset(dataset_name, split=split)
    assert isinstance(dataset, Dataset)
    _df = dataset.to_pandas()
    assert isinstance(_df, pd.DataFrame)

    # Filter dataset
    swe_bench_tests = filter_dataset(_df, "instance_id")
    logger.info(
        f"Loaded dataset {dataset_name} with split {split}: {len(swe_bench_tests)} tasks"
    )

    # Prepare dataset (apply n_limit if specified)
    instances = prepare_dataset(swe_bench_tests, output_file, eval_n_limit)
    return instances
