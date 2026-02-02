from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from benchmarks.swtbench.config import EVAL_DEFAULTS
from openhands.sdk import get_logger


logger = get_logger(__name__)


def ensure_swt_bench_repo(cache_dir: Path | None = None) -> Path:
    """
    Ensure the SWT-bench sources are available locally.

    Returns the repository path under the cache directory.
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "openhands" / "swt-bench"
    swt_bench_dir = cache_dir / "swt-bench"

    if swt_bench_dir.exists():
        return swt_bench_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning SWT-Bench repository into %s", swt_bench_dir)
    result = subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/logic-star-ai/swt-bench.git",
            str(swt_bench_dir),
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        logger.error("Failed to clone swt-bench: %s", result.stderr)
        raise RuntimeError("Unable to clone swt-bench repository")

    return swt_bench_dir


def _load_instance_ids(output_jsonl: Path) -> list[str]:
    instance_ids: list[str] = []
    seen = set()
    with output_jsonl.open("r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping invalid JSON on line %s", line_num)
                continue
            instance_id = data.get("instance_id")
            if not instance_id or instance_id in seen:
                continue
            seen.add(instance_id)
            instance_ids.append(instance_id)
    return instance_ids


def compute_required_images(
    output_jsonl: Path,
    dataset: str,
    split: str,
) -> tuple[set[str], set[str]]:
    """
    Compute the base/env image tags required to evaluate the given predictions file.

    Returns (base_image_tags, env_image_tags).
    """
    instance_ids = _load_instance_ids(output_jsonl)
    if not instance_ids:
        raise ValueError(f"No instance_ids found in {output_jsonl}")

    swt_bench_dir = ensure_swt_bench_repo()
    sys.path.insert(0, str(swt_bench_dir / "src"))
    sys.path.insert(0, str(swt_bench_dir))

    # Delay import until after sys.path manipulation so we use the cached checkout.
    from src.dataset import load_swebench_dataset  # type: ignore[import-not-found]
    from src.exec_spec import make_exec_spec  # type: ignore[import-not-found]

    # Change to swt-bench directory for dataset loading (required for filter files)
    cwd = os.getcwd()
    try:
        os.chdir(swt_bench_dir)
        dataset_entries = load_swebench_dataset(
            name=dataset, split=split, is_swt=True, filter_swt=True
        )
    finally:
        os.chdir(cwd)
    entries_by_id = {entry["instance_id"]: entry for entry in dataset_entries}

    missing = [iid for iid in instance_ids if iid not in entries_by_id]
    if missing:
        logger.warning(
            "Predictions reference %s instance_ids not present in dataset: %s",
            len(missing),
            ", ".join(missing[:5]),
        )

    specs = [
        make_exec_spec(entries_by_id[iid])
        for iid in instance_ids
        if iid in entries_by_id
    ]
    if not specs:
        raise RuntimeError("No ExecSpecs produced; cannot compute required images.")

    base_images = {spec.base_image_key for spec in specs}
    env_images = {spec.env_image_key for spec in specs}
    logger.info(
        "Computed %s base images and %s env images for %s instances",
        len(base_images),
        len(env_images),
        len(specs),
    )
    return base_images, env_images


def format_images_plain(images: Iterable[str]) -> str:
    return "\n".join(sorted(images))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="List SWT-bench base/env images required for a predictions file."
    )
    parser.add_argument("output_jsonl", type=Path, help="Path to output.jsonl")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--split", help="Dataset split")
    parser.set_defaults(**EVAL_DEFAULTS)
    parser.add_argument(
        "--format",
        choices=["plain", "json"],
        default="plain",
        help="Output format",
    )
    args = parser.parse_args()

    base_images, env_images = compute_required_images(
        args.output_jsonl,
        args.dataset,
        args.split,
    )
    payload = {
        "base": sorted(base_images),
        "env": sorted(env_images),
    }

    if args.format == "json":
        print(json.dumps(payload))
    else:
        print(format_images_plain(payload["base"] + payload["env"]))


if __name__ == "__main__":
    # Configure root logging for ad-hoc usage
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
