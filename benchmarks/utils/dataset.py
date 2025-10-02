from __future__ import annotations

import inspect
import os
from typing import Optional

import pandas as pd
import toml
from datasets import Dataset, load_dataset

from openhands.sdk import get_logger


logger = get_logger(__name__)


def filter_dataset(
    dataset: pd.DataFrame, filter_column: str, config_path: Optional[str] = None
) -> pd.DataFrame:
    """Filter dataset based on config.toml selected_ids and environment variables."""

    # If config_path is not provided, try to determine it from the calling context
    if config_path is None:
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find the benchmark directory
            caller_frame = frame.f_back if frame else None
            while caller_frame:
                caller_file = caller_frame.f_code.co_filename

                # Look for benchmark directory in the path - prioritize run_infer.py
                if "/benchmarks/" in caller_file and "run_infer.py" in caller_file:
                    # Extract benchmark directory from path
                    # e.g., /path/to/benchmarks/swe_bench/run_infer.py -> swe_bench
                    path_parts = caller_file.split("/")
                    try:
                        benchmarks_idx = path_parts.index("benchmarks")
                        if benchmarks_idx + 1 < len(path_parts):
                            config_path = os.path.join(
                                os.path.dirname(caller_file), "config.toml"
                            )
                            break
                    except ValueError:
                        pass  # 'benchmarks' not found in path
                caller_frame = caller_frame.f_back
        finally:
            del frame

    # Try to read config.toml if we found a path
    if config_path and os.path.exists(config_path):
        try:
            config = toml.load(config_path)
            selected_ids = config.get("selected_ids", [])

            if selected_ids:
                logger.info(
                    f"Filtering by {len(selected_ids)} selected_ids from {config_path}"
                )
                # Filter by selected instance IDs first
                if filter_column in dataset.columns:
                    original_size = len(dataset)
                    # pandas boolean should return a DataFrame
                    dataset = dataset[dataset[filter_column].isin(selected_ids)]  # type: ignore
                    logger.info(
                        f"Filtered {original_size} to {len(dataset)} selected_ids"
                    )
                else:
                    logger.warning(
                        f"Filter column '{filter_column}' not found in dataset"
                    )
            else:
                logger.info(
                    f"No selected_ids found in {config_path}, using full dataset"
                )

        except Exception as e:
            logger.warning(f"Failed to read config.toml from {config_path}: {e}")
    elif config_path:
        logger.info(f"No config.toml found at {config_path}, using full dataset")
    else:
        logger.info("No config.toml path determined, using full dataset")

    # Apply any additional environment variable based filtering here
    # (placeholder for future enhancements)

    return dataset


def prepare_dataset(
    dataset: pd.DataFrame,
    output_file: str,
    n_limit: int,
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
    if n_limit > 0:
        dataset = dataset.head(n_limit)

    return dataset


def get_dataset(
    dataset_name: str,
    split: str,
    output_file: str,
    eval_n_limit: int,
    completed_instances: Optional[set] = None,
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
        f"Loaded dataset {dataset_name} with split {split}: "
        f"{len(swe_bench_tests)} tasks"
    )

    # Prepare dataset (apply n_limit if specified and filter completed)
    instances = prepare_dataset(
        swe_bench_tests, output_file, eval_n_limit, completed_instances
    )
    return instances
