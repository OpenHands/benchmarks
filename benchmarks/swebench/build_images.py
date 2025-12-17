#!/usr/bin/env python3
"""
Build agent-server images for all unique SWE-Bench base images in a dataset split,
then wrap them with a lightweight layer that pins docutils<0.21 and installs roman.

Example:
  uv run benchmarks/swebench/build_images.py \
    --dataset princeton-nlp/SWE-bench_Verified --split test \
    --image ghcr.io/openhands/eval-agent-server --target source-minimal
"""

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from benchmarks.utils.build_utils import (
    BuildOutput,
    _update_pbar,
    build_all_images,
    capture_output,
    default_build_output_dir,
    get_build_parser,
)
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.image_utils import image_exists
from benchmarks.utils.version import SDK_SHORT_SHA
from openhands.sdk import get_logger

logger = get_logger(__name__)
WRAPPER_DOCKERFILE = Path(__file__).with_name("Dockerfile.swebench-deps")
# Repos that require the docutils/roman wrapper layer
WRAPPED_REPOS = {"sphinx-doc"}


def get_official_docker_image(
    instance_id: str,
    docker_image_prefix="docker.io/swebench/",
) -> str:
    # Official SWE-Bench image
    # swebench/sweb.eval.x86_64.django_1776_django-11333:v1
    repo, name = instance_id.split("__")
    official_image_name = docker_image_prefix.rstrip("/")
    official_image_name += f"/sweb.eval.x86_64.{repo}_1776_{name}:latest".lower()
    logger.debug(f"Official SWE-Bench image: {official_image_name}")
    return official_image_name


def extract_custom_tag(base_image: str) -> str:
    """
    Extract SWE-Bench instance ID from official SWE-Bench image name.

    Example:
        docker.io/swebench/sweb.eval.x86_64.django_1776_django-12155:latest
        -> sweb.eval.x86_64.django_1776_django-12155
    """
    name_tag = base_image.split("/")[-1]
    name = name_tag.split(":")[0]
    return name


def should_wrap_custom_tag(custom_tag: str) -> bool:
    prefix = "sweb.eval.x86_64."
    if custom_tag.startswith(prefix):
        custom_tag = custom_tag[len(prefix) :]
    return custom_tag.split("_", 1)[0] in WRAPPED_REPOS


def should_wrap_instance_id(instance_id: str) -> bool:
    repo = instance_id.split("__")[0]
    return repo in WRAPPED_REPOS


def collect_unique_base_images(
    dataset,
    split,
    n_limit,
    selected_instances_file: str | None = None,
):
    df = get_dataset(
        dataset_name=dataset,
        split=split,
        eval_limit=n_limit if n_limit else None,
        selected_instances_file=selected_instances_file,
    )
    return sorted(
        {get_official_docker_image(str(row["instance_id"])) for _, row in df.iterrows()}
    )


def _target_suffix(target: str) -> str:
    """Mirror the tagging convention used by run_infer."""
    return "" if target == "binary" else f"-{target}"


def _agent_server_tag(
    image_repo: str, custom_tag: str, target: str, sdk_short_sha: str
) -> str:
    suffix = _target_suffix(target)
    return f"{image_repo}:{sdk_short_sha}-{custom_tag}{suffix}"


def build_wrapped_image(base_agent_image: str, push: bool = False) -> BuildOutput:
    """
    Build a single wrapped image and return its BuildOutput. Used for local runs.
    """
    return _wrap_image(base_agent_image, push)


def _wrap_image(base_agent_image: str, push: bool) -> BuildOutput:
    """
    Wrap an agent-server image with pinned docutils/roman.

    For pushes, verify the base tag exists in the registry; for local loads,
    let Docker resolve the base image locally.
    """
    if push and not image_exists(base_agent_image):
        return BuildOutput(
            base_image=base_agent_image,
            tags=[],
            error=(
                f"Base agent-server image {base_agent_image} not found in registry. "
                "Build and push it before wrapping."
            ),
        )

    if not WRAPPER_DOCKERFILE.exists():
        return BuildOutput(
            base_image=base_agent_image,
            tags=[],
            error=f"Wrapper Dockerfile not found at {WRAPPER_DOCKERFILE}",
        )

    args = [
        "docker",
        "buildx",
        "build",
        "--file",
        str(WRAPPER_DOCKERFILE),
        "--build-arg",
        f"SDK_IMAGE={base_agent_image}",
        "--tag",
        base_agent_image,
    ]
    if push:
        args += ["--platform", "linux/amd64", "--push"]
    else:
        args += ["--load"]
    args.append(str(WRAPPER_DOCKERFILE.parent))

    logger.info("Wrapping %s in-place", base_agent_image)
    proc = subprocess.run(args, text=True, capture_output=True)

    # Stream captured output so callers still see build logs
    sys.stdout.write(proc.stdout or "")
    sys.stderr.write(proc.stderr or "")

    if proc.returncode != 0:
        error = (
            proc.stderr.strip()
            or proc.stdout.strip()
            or f"Wrapper build failed with exit code {proc.returncode}"
        )
        return BuildOutput(base_image=base_agent_image, tags=[], error=error)

    return BuildOutput(base_image=base_agent_image, tags=[base_agent_image], error=None)


def _wrap_with_logging(
    log_dir: Path,
    base_agent_image: str,
    push: bool,
    max_retries: int,
) -> BuildOutput:
    """
    Build a wrapped image with retry + streamed logging to the build log directory.
    """
    safe_name = base_agent_image.replace("/", "_").replace(":", "_")

    for attempt in range(max_retries):
        with capture_output(safe_name, log_dir) as log_path:
            if attempt:
                logger.info(
                    "Retrying wrapper build for %s (%s/%s)",
                    base_agent_image,
                    attempt + 1,
                    max_retries,
                )
            result = _wrap_image(base_agent_image, push)
            result.log_path = str(log_path)
            if result.error is None:
                return result

        if attempt == max_retries - 1:
            return result

    raise RuntimeError("Unreachable: wrapper retries exhausted")


def wrap_all_images(
    base_agent_images: list[str],
    build_dir: Path,
    push: bool,
    max_workers: int,
    max_retries: int,
) -> int:
    """
    Wrap all agent-server images with the docutils/roman dependency layer.
    """
    log_dir = build_dir / "logs-wrapped"
    manifest_path = build_dir / "manifest-wrapped.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures = 0
    in_progress: set[str] = set()

    with (
        manifest_path.open("w") as writer,
        tqdm(
            total=len(base_agent_images),
            desc="Wrapping SWE-bench images",
            leave=True,
        ) as pbar,
    ):
        _update_pbar(pbar, successes, failures, 0, None, "Queueing")

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for base in base_agent_images:
                in_progress.add(base)
                fut = ex.submit(_wrap_with_logging, log_dir, base, push, max_retries)
                futures[fut] = base

            _update_pbar(
                pbar,
                successes,
                failures,
                len(in_progress),
                next(iter(in_progress), None),
                "Running",
            )

            for fut in as_completed(futures):
                base = futures[fut]
                try:
                    result: BuildOutput = fut.result()
                    writer.write(result.model_dump_json() + "\n")
                    writer.flush()
                    if result.error:
                        failures += 1
                        _update_pbar(
                            pbar,
                            successes,
                            failures,
                            len(in_progress),
                            base,
                            "❌ Failed",
                        )
                    else:
                        successes += 1
                        _update_pbar(
                            pbar, successes, failures, len(in_progress), base, "✅ Done"
                        )
                except Exception as e:
                    failures += 1
                    writer.write(
                        BuildOutput(
                            base_image=base, tags=[], error=repr(e)
                        ).model_dump_json()
                        + "\n"
                    )
                    writer.flush()
                    logger.error("Wrapper build failed for %s: %r", base, e)
                    _update_pbar(
                        pbar, successes, failures, len(in_progress), base, "❌ Failed"
                    )
                finally:
                    in_progress.discard(base)
                    pbar.update(1)
                    _update_pbar(
                        pbar,
                        successes,
                        failures,
                        len(in_progress),
                        next(iter(in_progress), None),
                        None,
                    )

    logger.info(
        "Wrapper builds complete. Built=%d Failed=%d Manifest=%s",
        successes,
        failures,
        str(manifest_path),
    )
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    parser = get_build_parser()
    args = parser.parse_args(argv)

    base_images: list[str] = collect_unique_base_images(
        args.dataset,
        args.split,
        args.n_limit,
        args.select,
    )
    build_dir = default_build_output_dir(args.dataset, args.split)

    base_agent_entries = [
        (
            _agent_server_tag(
                args.image, extract_custom_tag(base), args.target, SDK_SHORT_SHA
            ),
            extract_custom_tag(base),
        )
        for base in base_images
    ]

    base_agent_images = [img for img, _ in base_agent_entries]
    wrapped_agent_images = [
        img
        for img, custom_tag in base_agent_entries
        if should_wrap_custom_tag(custom_tag)
    ]

    rc = build_all_images(
        base_images=base_images,
        target=args.target,
        build_dir=build_dir,
        image=args.image,
        push=args.push,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        base_image_to_custom_tag_fn=extract_custom_tag,
    )
    if args.dry_run:
        # build_all_images already printed base images; also show wrapped tags
        if wrapped_agent_images:
            print("\n".join(wrapped_agent_images))
        return rc

    wrap_rc = 0
    if wrapped_agent_images:
        wrap_rc = wrap_all_images(
            base_agent_images=wrapped_agent_images,
            build_dir=build_dir,
            push=args.push,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
        )
    else:
        logger.info("No instances require wrapper layer; skipping wrap stage.")

    return 1 if rc or wrap_rc else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
