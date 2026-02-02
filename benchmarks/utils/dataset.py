from __future__ import annotations

import os
import time
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
) -> pd.DataFrame:
    """Prepare dataset for evaluation."""

    # Filter to selected instances first (if provided)
    if selected_instances_file:
        selected_instances = _load_selected_instances(selected_instances_file)
        original_size = len(dataset)
        mask = dataset["instance_id"].isin(list(selected_instances))
        dataset = cast(pd.DataFrame, dataset[mask])
        logger.info(
            f"Selected {len(dataset)} instances from {original_size} total instances"
        )

    # Apply limit after filtering completed instances
    if n_limit is not None and n_limit > 0:
        dataset = dataset.sample(n=min(n_limit, len(dataset)), random_state=42)

    return dataset


def _load_hf_dataset_with_retry(dataset_name: str, split: str) -> Dataset:
    """Load a Hugging Face dataset with retries and longer HTTP timeouts."""
    # Default HF timeout is ~10s; bump it to reduce transient ReadTimeouts.
    os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", os.environ["HF_HUB_HTTP_TIMEOUT"])

    attempts = 5
    backoff = 5.0
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            dataset = load_dataset(dataset_name, split=split)
            assert isinstance(dataset, Dataset)
            return dataset
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                break
            wait = min(backoff, 60.0)
            logger.warning(
                "load_dataset failed (attempt %s/%s): %s; retrying in %.1fs",
                attempt,
                attempts,
                exc,
                wait,
            )
            time.sleep(wait)
            backoff *= 2

    assert last_exc is not None
    raise last_exc


def get_dataset(
    dataset_name: str,
    split: str,
    eval_limit: int | None = None,
    selected_instances_file: str | None = None,
) -> pd.DataFrame:
    """Load and prepare dataset for evaluation."""
    # Check if dataset_name is a local file path
    if os.path.isfile(dataset_name) and dataset_name.endswith(".jsonl"):
        # Load local JSONL file
        dataset = load_dataset("json", data_files=dataset_name, split="train")
        assert isinstance(dataset, Dataset)
        df = dataset.to_pandas()
        assert isinstance(df, pd.DataFrame)
    else:
        # Load dataset from HuggingFace Hub
        dataset = _load_hf_dataset_with_retry(dataset_name, split)
        df = dataset.to_pandas()
        assert isinstance(df, pd.DataFrame)

    # TODO: Add the ability to filter dataset
    logger.info(f"Loaded dataset {dataset_name} with split {split}: {len(df)} tasks")

    # Prepare dataset (apply n_limit if specified and filter selected)
    instances = prepare_dataset(df, eval_limit, selected_instances_file)
    return instances
