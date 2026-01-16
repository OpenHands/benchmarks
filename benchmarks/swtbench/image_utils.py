from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from benchmarks.utils.image_utils import image_exists
from openhands.sdk import get_logger


logger = get_logger(__name__)
DEFAULT_EVAL_IMAGE_PREFIX = "ghcr.io/openhands/swtbench-eval"


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
    *,
    filter_swt: bool = True,
    is_swt: bool = False,
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

    dataset_entries = load_swebench_dataset(
        name=dataset, split=split, is_swt=is_swt, filter_swt=filter_swt
    )
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


def _run_docker(cmd: list[str]) -> tuple[bool, str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or "").strip()
    return True, (result.stdout or "").strip()


def pull_prebaked_eval_images(
    predictions_file: Path,
    dataset: str,
    split: str,
    *,
    image_prefix: str | None = None,
    gh_username: str | None = None,
    gh_pat: str | None = None,
) -> tuple[bool, dict]:
    """
    Attempt to pull prebaked SWT-bench eval base/env images from a registry.

    Returns (all_available, details_dict).
    """
    prefix = (
        image_prefix
        or os.getenv("SWT_BENCH_EVAL_IMAGE_PREFIX")
        or DEFAULT_EVAL_IMAGE_PREFIX
    ).rstrip("/")
    details: dict = {
        "prefix": prefix,
        "dataset": dataset,
        "split": split,
    }

    if not prefix:
        details["error"] = "empty_prefix"
        return False, details

    try:
        base_images, env_images = compute_required_images(
            predictions_file, dataset, split
        )
    except Exception as exc:  # pragma: no cover - network/FS issues
        details["error"] = f"compute_failed: {exc}"
        return False, details

    required = sorted(base_images | env_images)
    details["required_count"] = len(required)
    if not required:
        details["error"] = "no_required_images"
        return False, details

    gh_user = gh_username or os.getenv("GHCR_USERNAME") or os.getenv("GITHUB_ACTOR")
    gh_token = gh_pat or os.getenv("GHCR_PAT")

    missing: list[dict] = []
    pulled: list[str] = []
    pull_errors: list[dict] = []

    for tag in required:
        remote_tag = f"{prefix}/{tag}"
        exists = image_exists(remote_tag, gh_username=gh_user, gh_pat=gh_token)
        if not exists:
            missing.append({"remote": remote_tag, "tag": tag, "reason": "not_found"})
            continue

        ok, err = _run_docker(["docker", "pull", remote_tag])
        if not ok:
            pull_errors.append(
                {
                    "remote": remote_tag,
                    "tag": tag,
                    "reason": "pull_failed",
                    "error": err,
                }
            )
            missing.append({"remote": remote_tag, "tag": tag, "reason": "pull_failed"})
            continue

        ok, err = _run_docker(["docker", "tag", remote_tag, tag])
        if not ok:
            pull_errors.append(
                {"remote": remote_tag, "tag": tag, "reason": "tag_failed", "error": err}
            )
            missing.append({"remote": remote_tag, "tag": tag, "reason": "tag_failed"})
            continue

        pulled.append(tag)

    details["missing"] = missing
    details["pulled"] = pulled
    details["pull_errors"] = pull_errors
    details["used_auth"] = bool(gh_user and gh_token)
    return len(missing) == 0, details


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="List SWT-bench base/env images required for a predictions file."
    )
    parser.add_argument("output_jsonl", type=Path, help="Path to output.jsonl")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument(
        "--no-filter-swt",
        action="store_true",
        help="Disable SWT filtering when loading the dataset",
    )
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
        filter_swt=not args.no_filter_swt,
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
