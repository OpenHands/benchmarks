from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import docker

from benchmarks.swtbench.image_utils import ensure_swt_bench_repo
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.image_utils import image_exists as remote_image_exists
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

    cwd = os.getcwd()
    try:
        os.chdir(swt_bench_dir)
        dataset_entries = load_swebench_dataset(
            name=dataset, split=split, is_swt=False, filter_swt=filter_swt
        )
    finally:
        os.chdir(cwd)
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


def build_env_images(
    exec_specs: list,
    max_workers: int,
    build_mode: str,
    max_retries: int,
    batch_size: int,
    image_prefix: str | None,
) -> None:
    """
    Build base + environment images required by the provided ExecSpecs.

    Images are pushed immediately after each successful build when image_prefix is set,
    so partial progress is kept if the workflow fails mid-run.
    """
    from src.docker_build import (  # type: ignore[import-not-found]
        BuildImageError,
        build_base_images,
        build_env_images as build_envs,
    )

    client = docker.from_env()
    total_base = len({spec.base_image_key for spec in exec_specs})
    total_env = len({spec.env_image_key for spec in exec_specs})
    remote_prefix = image_prefix.rstrip("/") if image_prefix else None

    base_to_build_keys: set[str] = set()

    def prefixed(tag: str) -> str | None:
        return f"{remote_prefix}/{tag}" if remote_prefix else None

    base_spec_by_key = {}
    for spec in exec_specs:
        key = spec.base_image_key
        base_spec_by_key.setdefault(key, spec)
        remote_tag = prefixed(key)

        if remote_tag and remote_image_exists(remote_tag):
            logger.info("Base image %s already in registry; reusing", remote_tag)
            try:
                img = client.images.pull(remote_tag)
                if remote_tag != key:
                    img.tag(key)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning(
                    "Failed to pull %s (%s); will rebuild locally", remote_tag, exc
                )
                base_to_build_keys.add(key)
                continue
            continue

        base_to_build_keys.add(key)

    missing_base_specs = [base_spec_by_key[k] for k in base_to_build_keys]
    skipped_base = total_base - len(base_to_build_keys)

    if missing_base_specs:
        logger.info(
            "Building %s/%s base images (skipping %s already present)",
            len({spec.base_image_key for spec in missing_base_specs}),
            total_base,
            skipped_base,
        )
        build_base_images(
            client, missing_base_specs, force_rebuild=False, build_mode=build_mode
        )
        base_built = {spec.base_image_key for spec in missing_base_specs}
        if image_prefix:
            tag_and_push(base_built, image_prefix)
    else:
        logger.info(
            "All %s base images already exist; skipping base builds", total_base
        )

    missing_env_specs: list = []

    for spec in exec_specs:
        key = spec.env_image_key
        remote_tag = prefixed(key)

        if remote_tag and remote_image_exists(remote_tag):
            logger.info("Env image %s already in registry; skipping build", remote_tag)
            continue

        missing_env_specs.append(spec)

    if not missing_env_specs:
        logger.info("All %s env images already exist; skipping env builds", total_env)
        return

    successful_images: list[str] = []
    failed_images: list[str] = []
    
    logger.info(
        "Building %s/%s env images (no timeout)",
        len({spec.env_image_key for spec in missing_env_specs}),
        total_env,
    )
    
    for idx, spec in enumerate(missing_env_specs, start=1):
        env_key = spec.env_image_key
        logger.info(
            "Building env image %s/%s: %s",
            idx,
            len(missing_env_specs),
            env_key,
        )
        
        start_time = time.time()
        try:
            build_envs(
                client,
                [spec],
                force_rebuild=False,
                max_workers=1,
                build_mode=build_mode,
            )
            elapsed = time.time() - start_time
            logger.info(
                "✅ Successfully built %s in %.1f min",
                env_key,
                elapsed / 60,
            )
            successful_images.append(env_key)
            # Push immediately after successful build
            if image_prefix:
                tag_and_push([env_key], image_prefix)
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "❌ Failed to build %s after %.1f min: %s",
                env_key,
                elapsed / 60,
                str(e),
            )
            failed_images.append(env_key)
    
    # Summary
    logger.info(
        "Build summary: %d successful, %d failed",
        len(successful_images),
        len(failed_images),
    )
    
    if failed_images:
        logger.warning(
            "Failed images: %s",
            ", ".join(failed_images),
        )
    
    return


def chunked(seq: Sequence, size: int) -> Iterator[List]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


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
        "--arch",
        choices=["x86_64", "arm64", ""],
        default="",
        help="Force architecture for built images (defaults to host arch)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel builds for env images",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per batch for env image builds",
    )
    parser.add_argument(
        "--build-batch-size",
        type=int,
        default=10,
        help="Number of env images to build per batch",
    )
    parser.add_argument(
        "--build-mode",
        choices=["api", "cli"],
        default="cli",
        help="swt-bench build mode",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Build images locally without pushing to the registry",
    )
    # Keep --image-timeout for backward compatibility but ignore it
    parser.add_argument(
        "--image-timeout",
        type=int,
        default=0,
        help="DEPRECATED: Timeout has been removed. Images build without timeout.",
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
    if args.arch:
        for spec in exec_specs:
            spec.arch = args.arch
        logger.info("Overrode ExecSpec architecture to %s", args.arch)

    build_env_images(
        exec_specs,
        max_workers=args.max_workers,
        build_mode=args.build_mode,
        max_retries=args.max_retries,
        batch_size=args.build_batch_size,
        image_prefix=None if args.no_push else args.image_prefix,
    )

    base_images = {spec.base_image_key for spec in exec_specs}
    env_images = {spec.env_image_key for spec in exec_specs}
    logger.info("Built images: %s base, %s env", len(base_images), len(env_images))

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "instances": target_ids,
        "base_images": sorted(base_images),
        "env_images": sorted(env_images),
        "image_prefix": args.image_prefix,
        "arch": args.arch or "host",
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
