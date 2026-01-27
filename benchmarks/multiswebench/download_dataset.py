"""
Download and process Multi-SWE-bench dataset from Hugging Face.

This module provides functionality to download the Multi-SWE-bench dataset
and concatenate instances by programming language.
"""

import json

from huggingface_hub import hf_hub_download, list_repo_files

from benchmarks.multiswebench.constants import DATASET_CACHE_DIR
from openhands.sdk import get_logger


logger = get_logger(__name__)


def download_and_concat_dataset(dataset_path: str, language: str) -> str:
    """
    Download Multi-SWE-bench dataset and concatenate instances by language.

    Args:
        dataset_path: HuggingFace dataset path (e.g., "ByteDance-Seed/Multi-SWE-bench")
        language: Programming language to filter by (e.g., "java", "python", "javascript")

    Returns:
        Path to the concatenated JSONL file containing all instances for the specified language

    Example:
        >>> path = download_and_concat_dataset("ByteDance-Seed/Multi-SWE-bench", "java")
        >>> print(f"Java instances saved to: {path}")
    """
    # Ensure cache directory exists
    DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Create output filename based on dataset and language
    dataset_name = dataset_path.replace("/", "_").replace("-", "_")
    output_filename = f"{dataset_name}_{language.lower()}.jsonl"
    output_path = DATASET_CACHE_DIR / output_filename

    # Check if file already exists
    if output_path.exists():
        logger.info(f"Dataset file already exists: {output_path}")
        return str(output_path)

    logger.info(f"Downloading dataset: {dataset_path}")

    # Download the dataset files directly from HuggingFace Hub
    try:
        # List all files in the repository
        repo_files = list_repo_files(dataset_path, repo_type="dataset")
        logger.info(f"Found {len(repo_files)} files in repository")

        # Filter for JSONL files
        jsonl_files = [f for f in repo_files if f.endswith(".jsonl")]
        logger.info(f"Found JSONL files: {jsonl_files}")

        logger.info(f"Found {len(jsonl_files)} JSONL files in repository")

    except Exception as e:
        logger.error(f"Failed to download dataset {dataset_path}: {e}")
        raise

    # Create a mapping from file paths to languages based on directory structure
    file_language_map = {}
    for file_path in jsonl_files:
        # Extract language from file path like "java/alibaba__fastjson2_dataset.jsonl"
        if "/" in file_path:
            file_language = file_path.split("/")[0].lower()
            file_language_map[file_path] = file_language

    # Filter instances by language
    language_instances = []
    language_lower = language.lower()

    # Process each file and its instances
    for file_path in jsonl_files:
        file_language = file_language_map.get(file_path, "").lower()

        # Skip files that don't match the target language
        if file_language != language_lower:
            continue

        logger.info(f"Processing {file_language} file: {file_path}")
        local_file = hf_hub_download(
            repo_id=dataset_path,
            filename=file_path,
            repo_type="dataset",
            cache_dir=str(DATASET_CACHE_DIR / "hf_cache"),
        )

        # Read the JSONL file and add all instances
        with open(local_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    instance = json.loads(line)
                    language_instances.append(instance)
                    logger.debug(
                        f"Added instance: {instance.get('instance_id', 'unknown')}"
                    )

    logger.info(f"Found {len(language_instances)} instances for language '{language}'")

    if not language_instances:
        logger.warning(
            f"No instances found for language '{language}'. Available languages might be different."
        )
        # Log available languages from file paths
        available_languages = set()
        for file_path in jsonl_files:
            if "/" in file_path:
                available_languages.add(file_path.split("/")[0].lower())
        logger.info(f"Available languages in dataset: {sorted(available_languages)}")

    # Write concatenated instances to JSONL file
    with open(output_path, "w", encoding="utf-8") as f:
        for instance in language_instances:
            f.write(json.dumps(instance) + "\n")

    logger.info(
        f"Saved {len(language_instances)} {language} instances to: {output_path}"
    )
    return str(output_path)
