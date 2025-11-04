#!/usr/bin/env python3
"""
Build agent-server images for all unique SWE-Bench base images in a dataset split.

Example:
  uv run benchmarks/swe_bench/build_images.py \
    --dataset princeton-nlp/SWE-bench_Verified --split test \
    --image ghcr.io/openhands/agent-server --target source-minimal
"""

import argparse
import contextlib
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from benchmarks.swe_bench.run_infer import get_official_docker_image
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.dataset import get_dataset
from openhands.agent_server.docker.build import BuildOptions, build
from openhands.sdk import get_logger


logger = get_logger(__name__)


@contextlib.contextmanager
def capture_output(base_name: str, out_dir: Path):
    """
    Capture stdout/stderr during a block and stream them to:
      <out_dir>/<base_name>/build-<timestamp>.log

    Keeps redirect_* semantics; writes are realtime (line-buffered + flush).
    Yields the log_path.
    """
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    log_path = Path(out_dir) / base_name / f"build-{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # tell the user where we’re logging, without being swallowed by the redirect
    # (goes to the original stderr so it’s visible immediately)
    logger.info(f"Logging build output to {log_path}")

    # Open line-buffered so writes flush on newlines;
    # also wrap to hard-flush every write.
    f = log_path.open("w", encoding="utf-8", buffering=1)

    class _FlushOnWrite(io.TextIOBase):
        encoding = f.encoding

        def __init__(self, sink):
            self._sink = sink

        def write(self, s):
            n = self._sink.write(s)
            self._sink.flush()
            return n

        def flush(self):
            self._sink.flush()

        def fileno(self):
            # allow libs that try to detect fileno()
            return self._sink.fileno()

    sink = _FlushOnWrite(f)

    # Redirect stdout/stderr to the same realtime sink.
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):  # type: ignore[arg-type]
        try:
            yield log_path
        finally:
            # make sure everything is on disk
            sink.flush()
            f.close()


def extend_parser() -> argparse.ArgumentParser:
    """Reuse benchmark parser and extend with build-related options."""
    parser = get_parser(add_llm_config=False)
    parser.description = "Build all agent-server images for SWE-Bench base images."

    parser.add_argument(
        "--docker-image-prefix",
        default="docker.io/swebench/",
        help="Prefix for SWE-Bench images",
    )
    parser.add_argument(
        "--image",
        default="ghcr.io/openhands/agent-server",
        help="Target repo/name for built image",
    )
    parser.add_argument(
        "--target",
        default="source-minimal",
        help="Build target (source | source-minimal | binary | binary-minimal)",
    )
    parser.add_argument(
        "--platforms", default="linux/amd64", help="Comma-separated platforms"
    )
    parser.add_argument("--custom-tags", default="", help="Comma-separated custom tags")
    parser.add_argument(
        "--push", action="store_true", help="Push via buildx instead of load locally"
    )
    parser.add_argument(
        "--max-workers", type=int, default=1, help="Concurrent builds (be cautious)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List base images only, don’t build"
    )
    return parser


def collect_unique_base_images(dataset, split, prefix, n_limit):
    df = get_dataset(
        dataset_name=dataset, split=split, eval_limit=n_limit if n_limit else None
    )
    return sorted(
        {
            get_official_docker_image(str(row["instance_id"]), prefix)
            for _, row in df.iterrows()
        }
    )


class BuildOutput(BaseModel):
    time: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    base_image: str
    tags: list[str]
    error: str | None = None
    log_path: str | None = None


def build_one(base_image: str, args: argparse.Namespace) -> BuildOutput:
    opts = BuildOptions(
        base_image=base_image,
        custom_tags=args.custom_tags,
        image=args.image,
        target=args.target,
        platforms=[p.strip() for p in args.platforms.split(",") if p.strip()],
        push=args.push,
    )
    tags = build(opts)
    return BuildOutput(base_image=base_image, tags=tags, error=None)


def _default_build_output_dir(
    dataset: str, split: str, base_dir: Path | None = None
) -> Path:
    """
    Default: ./builds/<dataset>/<split>
    Keeps build outputs in one predictable place, easy to .gitignore.
    """
    root = (base_dir or Path.cwd()) / "builds" / dataset / split
    root.mkdir(parents=True, exist_ok=True)
    return root


def _update_pbar(
    pbar: tqdm,
    successes: int,
    failures: int,
    running: int,
    sample: str | None,
    last_event: str | None,
):
    postfix = f"✅ {successes}  ❌ {failures}  🏃 {running}"
    if sample:
        postfix += f" ({sample})"
    if last_event:
        pbar.set_description(last_event)
    pbar.set_postfix_str(postfix, refresh=True)


def main(argv: list[str]) -> int:
    parser = extend_parser()
    args = parser.parse_args(argv)

    bases: list[str] = collect_unique_base_images(
        args.dataset, args.split, args.docker_image_prefix, args.n_limit
    )
    # Decide manifest path under ./builds/<dataset>/<split>/
    BUILD_DIR = _default_build_output_dir(args.dataset, args.split)
    BUILD_LOG_DIR = BUILD_DIR / "logs"
    manifest_path = BUILD_DIR / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def build_one_fn(base: str, args) -> BuildOutput:
        with capture_output(base, BUILD_LOG_DIR) as log_path:
            result = build_one(base, args)
            result.log_path = str(log_path)
            return result

    if args.dry_run:
        print("\n".join(bases))
        return 0

    successes = 0
    failures = 0
    in_progress: set[str] = set()
    mu = Lock()

    with (
        manifest_path.open("w") as writer,
        tqdm(total=len(bases), desc="Building agent-server images", leave=True) as pbar,
    ):
        _update_pbar(pbar, successes, failures, 0, None, "Queueing")

        # Single unified path: ThreadPoolExecutor( max_workers = args.max_workers ),
        # even if it's 1
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {}
            for base in bases:
                in_progress.add(base)
                fut = ex.submit(build_one_fn, base, args)
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
                    with mu:
                        successes += 1
                    _update_pbar(
                        pbar, successes, failures, len(in_progress), base, "✅ Done"
                    )
                except Exception as e:
                    logger.error("Build failed for %s: %r", base, e)
                    # Write a failure line to manifest; keep going.
                    writer.write(
                        BuildOutput(
                            base_image=base, tags=[], error=repr(e)
                        ).model_dump_json()
                        + "\n"
                    )
                    writer.flush()
                    with mu:
                        failures += 1
                    _update_pbar(
                        pbar, successes, failures, len(in_progress), base, "❌ Failed"
                    )
                finally:
                    with mu:
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

    # Optional: write a tiny summary JSON next to the manifest for quick reads
    summary_path = manifest_path.with_name("summary.json")
    summary_path.write_text(
        (
            "{"
            f'"dataset":"{args.dataset}",'
            f'"split":"{args.split}",'
            f'"total_unique_base_images":{len(bases)},'
            f'"built":{successes},'
            f'"failed":{failures}'
            "}"
        ),
        encoding="utf-8",
    )

    logger.info(
        "Done. Built=%d  Failed=%d  Manifest=%s  Summary=%s",
        successes,
        failures,
        str(manifest_path),
        str(summary_path),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
