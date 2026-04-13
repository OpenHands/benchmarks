#!/usr/bin/env python3
"""
Prefetch the GAIA dataset from Hugging Face into a local cache.

Usage:
    HF_TOKEN=... python prefetch_gaia.py [--dest /path/to/cache] [--workers 4] [--max-retries 5]

Designed to be run once at image-build time (or before gaia-infer in CI), so
that the eval pod never needs to call hf_hub_download at runtime. This avoids:
  - the hf-xet token-refresh deadlock (OpenHands/benchmarks#658)
  - transient ChunkedEncodingError / IncompleteRead during streaming downloads

After a successful run, point HF_HOME / HF_DATASETS_CACHE at --dest in the
eval pod and gaia-infer will load entirely from disk.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ID = "gaia-benchmark/GAIA"
REPO_TYPE = "dataset"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dest",
        default=os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
        help="HF cache root to populate (default: $HF_HOME or ~/.cache/huggingface)",
    )
    p.add_argument("--workers", type=int, default=4, help="Concurrent download workers (default: 4)")
    p.add_argument("--max-retries", type=int, default=5, help="Retries on transient failures (default: 5)")
    p.add_argument(
        "--verify",
        action="store_true",
        help="After download, try datasets.load_dataset() on every level to verify completeness.",
    )
    return p.parse_args()


def configure_env(dest: str) -> None:
    """Force the legacy HTTP path (no xet) and point HF cache at --dest."""
    os.environ["HF_HOME"] = dest
    os.environ["HF_DATASETS_CACHE"] = str(Path(dest) / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(dest) / "hub")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    Path(dest).mkdir(parents=True, exist_ok=True)


def download_with_retries(workers: int, max_retries: int) -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("ERROR: GAIA is gated; set HF_TOKEN (an HF token with GAIA access).")

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        backoff = min(60, 5 * 2 ** (attempt - 1))
        print(f"[prefetch] attempt {attempt}/{max_retries} (workers={workers})", flush=True)
        try:
            local_dir = snapshot_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=token,
                max_workers=workers,
                etag_timeout=30,
                resume_download=True,
            )
            print(f"[prefetch] OK -> {local_dir}", flush=True)
            return local_dir
        except (HfHubHTTPError, OSError, Exception) as e:  # noqa: BLE001 - intentional broad catch
            last_err = e
            msg = f"{type(e).__name__}: {e}"
            # Don't retry on hard auth/gating errors
            if "401" in msg or "403" in msg or "gated" in msg.lower():
                sys.exit(f"ERROR: auth/gate failure, not retrying: {msg}")
            print(f"[prefetch] attempt {attempt} failed: {msg}", flush=True)
            if attempt < max_retries:
                print(f"[prefetch] sleeping {backoff}s before retry", flush=True)
                time.sleep(backoff)

    sys.exit(f"ERROR: snapshot_download failed after {max_retries} attempts: {last_err}")


def verify(local_dir: str) -> None:
    """Sanity-check that every level can be loaded from the local cache."""
    from datasets import load_dataset

    for level in ("2023_level1", "2023_level2", "2023_level3", "2023_all"):
        for split in ("validation", "test"):
            try:
                ds = load_dataset(local_dir, level, split=split)
                print(f"[verify] {level}/{split}: {len(ds)} rows OK", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[verify] {level}/{split}: FAILED — {type(e).__name__}: {e}", flush=True)


def main() -> int:
    args = parse_args()
    configure_env(args.dest)
    free_gb = shutil.disk_usage(args.dest).free / 1024**3
    print(f"[prefetch] dest={args.dest} free={free_gb:.1f} GiB", flush=True)
    if free_gb < 2:
        sys.exit("ERROR: less than 2 GiB free at destination; aborting.")

    t0 = time.time()
    local_dir = download_with_retries(args.workers, args.max_retries)
    print(f"[prefetch] download done in {time.time() - t0:.1f}s", flush=True)

    if args.verify:
        verify(local_dir)

    print(f"[prefetch] done. Set in your eval pod: HF_HOME={args.dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
