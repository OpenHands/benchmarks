from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import docker

from benchmarks.swtbench.image_utils import (
    ensure_swt_bench_repo,
    patch_swt_bench_for_micromamba,
)
from benchmarks.utils.dataset import get_dataset
from openhands.sdk import get_logger


logger = get_logger(__name__)


def select_instance_ids(
    dataset: str,
    split: str,
    eval_limit: int | None,
    selected_instances_file: str | None,
    instance_ids: list[str] | None,
) -> list[str]:
    """
    Select the instance IDs that match the inference sampling logic.
    """
    if instance_ids:
        return instance_ids

    df = get_dataset(
        dataset_name=dataset,
        split=split,
        eval_limit=eval_limit,
        selected_instances_file=selected_instances_file,
    )
    ids = df["instance_id"].tolist()
    if not ids:
        raise RuntimeError("No instances selected for image build.")
    logger.info("Selected %s instances for image build", len(ids))
    return ids


def load_exec_specs(
    swt_bench_dir: Path,
    dataset: str,
    split: str,
    instance_ids: Iterable[str],
    filter_swt: bool = True,
) -> list:
    """
    Load ExecSpec objects for the provided instance IDs.
    """
    sys.path.insert(0, str(swt_bench_dir / "src"))
    sys.path.insert(0, str(swt_bench_dir))
    from src.dataset import load_swebench_dataset  # type: ignore[import-not-found]
    from src.exec_spec import make_exec_spec  # type: ignore[import-not-found]

    dataset_entries = load_swebench_dataset(
        name=dataset, split=split, is_swt=False, filter_swt=filter_swt
    )
    by_id = {entry["instance_id"]: entry for entry in dataset_entries}

    specs = []
    missing = []
    for iid in instance_ids:
        if iid not in by_id:
            missing.append(iid)
            continue
        specs.append(make_exec_spec(by_id[iid]))

    if missing:
        logger.warning(
            "Skipped %s missing instance_ids not found in dataset: %s",
            len(missing),
            ", ".join(missing[:5]),
        )
    if not specs:
        raise RuntimeError("No ExecSpecs available after filtering instance IDs.")
    return specs


def build_env_images(exec_specs: list, max_workers: int, build_mode: str) -> None:
    """
    Build base + environment images required by the provided ExecSpecs.
    """
    from src.docker_build import (  # type: ignore[import-not-found]
        build_base_images,
        build_env_images as build_envs,
    )

    client = docker.from_env()
    logger.info(
        "Building %s base images and %s env images (mode=%s, workers=%s)",
        len({spec.base_image_key for spec in exec_specs}),
        len({spec.env_image_key for spec in exec_specs}),
        build_mode,
        max_workers,
    )
    build_base_images(client, exec_specs, force_rebuild=False, build_mode=build_mode)
    build_envs(
        client,
        exec_specs,
        force_rebuild=False,
        max_workers=max_workers,
        build_mode=build_mode,
    )


def tag_and_push(images: Iterable[str], prefix: str) -> list[str]:
    """
    Tag the provided images with the registry prefix and push them.
    """
    pushed: list[str] = []
    prefix = prefix.rstrip("/")
    for image in images:
        target = f"{prefix}/{image}"
        logger.info("Pushing %s -> %s", image, target)
        subprocess_run(["docker", "tag", image, target])
        subprocess_run(["docker", "push", target])
        pushed.append(target)
    return pushed


def subprocess_run(cmd: list[str]) -> None:
    import subprocess

    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        logger.error("Command failed (%s): %s", " ".join(cmd), result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and push prebaked SWT-bench eval env images."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=1,
        help="Match inference sampling by limiting instances (0 to disable)",
    )
    parser.add_argument(
        "--instance-ids",
        default="",
        help="Comma-separated instance IDs to force (overrides eval-limit)",
    )
    parser.add_argument(
        "--selected-instances-file",
        default="",
        help="Optional selected instances file used during inference",
    )
    parser.add_argument(
        "--image-prefix",
        default="ghcr.io/openhands/swtbench-eval",
        help="Registry prefix for pushed images",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel builds for env images",
    )
    parser.add_argument(
        "--build-mode",
        choices=["api", "cli"],
        default="api",
        help="swt-bench build mode",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Build images locally without pushing to the registry",
    )
    args = parser.parse_args()

    instance_ids = (
        [iid for iid in args.instance_ids.split(",") if iid]
        if args.instance_ids
        else None
    )
    eval_limit = None if instance_ids else args.eval_limit
    selected_file = args.selected_instances_file or None

    swt_bench_dir = ensure_swt_bench_repo()
    patch_swt_bench_for_micromamba(swt_bench_dir)

    target_ids = select_instance_ids(
        dataset=args.dataset,
        split=args.split,
        eval_limit=eval_limit,
        selected_instances_file=selected_file,
        instance_ids=instance_ids,
    )
    exec_specs = load_exec_specs(
        swt_bench_dir, args.dataset, args.split, target_ids, filter_swt=True
    )

    build_env_images(
        exec_specs, max_workers=args.max_workers, build_mode=args.build_mode
    )

    base_images = {spec.base_image_key for spec in exec_specs}
    env_images = {spec.env_image_key for spec in exec_specs}
    logger.info("Built images: %s base, %s env", len(base_images), len(env_images))

    if not args.no_push:
        pushed = tag_and_push(base_images | env_images, args.image_prefix)
        logger.info("Pushed %s images", len(pushed))

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "instances": target_ids,
        "base_images": sorted(base_images),
        "env_images": sorted(env_images),
        "image_prefix": args.image_prefix,
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
