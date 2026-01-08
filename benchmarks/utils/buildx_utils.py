#!/usr/bin/env python3
"""
Buildx/BuildKit utilities for image build resets and pruning.
"""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from openhands.sdk import get_logger


logger = get_logger(__name__)


def _read_reset_state(path: Path) -> dict[str, float]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _write_reset_state(path: Path, state: dict[str, float]) -> None:
    try:
        path.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass


def _last_reset_for_kind(kind: str, state: dict[str, float]) -> float:
    if kind == "full":
        return state.get("full", 0.0)
    if kind == "partial":
        return max(state.get("partial", 0.0), state.get("full", 0.0))
    return max(state.values(), default=0.0)


def _should_throttle_reset(
    kind: str, state: dict[str, float], now: float, throttle_sec: int
) -> bool:
    if throttle_sec <= 0:
        return False
    last = _last_reset_for_kind(kind, state)
    return last > 0 and (now - last) < throttle_sec


def _buildkit_prune_filters(
    base_image: str | None, target_image: str | None
) -> list[str]:
    patterns = []
    for value in (base_image, target_image):
        if value:
            patterns.append(re.escape(value))
    if not patterns:
        return []
    pattern = "|".join(patterns)
    return ["--filter", f"description~={pattern}"]


def reset_buildkit(
    reset_kind: str, base_image: str | None, target_image: str | None
) -> None:
    if os.getenv("BUILDKIT_RESET_ON_FAILURE", "1") == "0":
        return
    if reset_kind not in {"restart", "partial", "full"}:
        return

    lock_path = Path(os.getenv("BUILDKIT_RESET_LOCK", "/tmp/buildkit-reset.lock"))
    state_path = Path(
        os.getenv("BUILDKIT_RESET_STATE", "/tmp/buildkit-reset-state.json")
    )
    throttle_sec = int(os.getenv("BUILDKIT_RESET_THROTTLE_SEC", "300"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    lock_file = lock_path.open("w")
    try:
        try:
            import fcntl

            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except Exception:
            # Best-effort locking; continue without it on unsupported platforms.
            pass

        now = time.time()
        state = _read_reset_state(state_path)
        if _should_throttle_reset(reset_kind, state, now, throttle_sec):
            last = _last_reset_for_kind(reset_kind, state)
            logger.info(
                "Skipping buildx %s reset; last reset %.0fs ago",
                reset_kind,
                now - last,
            )
            return

        prune_filters = _buildkit_prune_filters(base_image, target_image)
        if reset_kind == "restart":
            cmds = [["docker", "buildx", "inspect", "--bootstrap"]]
        elif reset_kind == "partial":
            cmds = [
                ["docker", "buildx", "prune", "--force", *prune_filters],
                ["docker", "buildx", "inspect", "--bootstrap"],
            ]
        else:
            cmds = [
                ["docker", "buildx", "prune", "--all", "--force", *prune_filters],
                ["docker", "buildx", "inspect", "--bootstrap"],
            ]

        logger.warning(
            "Resetting buildx (%s) after BuildKit failure%s",
            reset_kind,
            f" with filters {prune_filters}" if prune_filters else "",
        )
        for cmd in cmds:
            proc = subprocess.run(cmd, text=True, capture_output=True)
            if proc.stdout:
                logger.info(proc.stdout.strip())
            if proc.stderr:
                logger.warning(proc.stderr.strip())

        state[reset_kind] = now
        _write_reset_state(state_path, state)
    finally:
        try:
            lock_file.close()
        except Exception:
            pass


def maybe_reset_buildkit(
    base_image: str, target_image: str, attempt: int, max_retries: int
) -> None:
    if attempt >= max_retries - 1:
        return
    if attempt == 0:
        reset_buildkit("partial", base_image, target_image)
    else:
        reset_buildkit("full", base_image, target_image)


def buildkit_disk_usage(root: str | Path = "/var/lib/buildkit") -> tuple[int, int]:
    """
    Return (used_bytes, total_bytes) for the BuildKit root. Missing path -> (0, 0).
    """
    path = Path(root)
    try:
        usage = shutil.disk_usage(path)
        return usage.used, usage.total
    except FileNotFoundError:
        logger.warning("BuildKit root %s not found when checking disk usage", path)
    except Exception as e:
        logger.warning("Unable to read disk usage for %s: %s", path, e)
    return 0, 0


def prune_buildkit_cache(
    keep_storage_gb: int | None = None,
    filters: list[str] | None = None,
) -> None:
    """
    Run docker buildx prune to free space on the BuildKit cache.
    keep_storage_gb: amount of cache to keep (pass None to keep default behavior).
    filters: optional list of buildx prune --filter values.
    """
    base_cmd = ["docker", "buildx", "prune", "--all", "--force"]

    def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        logger.info("Pruning BuildKit cache: %s", " ".join(cmd))
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.stdout:
            logger.info(proc.stdout.strip())
        if proc.stderr:
            logger.warning(proc.stderr.strip())
        return proc

    # Prefer the newer --max-storage flag; fall back to --keep-storage if unsupported.
    storage_flag: list[str] = []
    fallback_flag: list[str] = []
    if keep_storage_gb is not None and keep_storage_gb > 0:
        storage_flag = ["--max-storage", f"{keep_storage_gb}g"]
        fallback_flag = ["--keep-storage", f"{keep_storage_gb}g"]

    filter_flags: list[str] = []
    if filters:
        for f in filters:
            filter_flags += ["--filter", f]

    proc = _run(base_cmd + storage_flag + filter_flags)
    if (
        proc.returncode != 0
        and fallback_flag
        and "--max-storage" in " ".join(storage_flag)
    ):
        if "unknown flag: --max-storage" in proc.stderr:
            proc = _run(base_cmd + fallback_flag + filter_flags)

    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr.strip()
            or proc.stdout.strip()
            or f"docker buildx prune failed with exit code {proc.returncode}"
        )


def maybe_prune_buildkit_cache(
    keep_storage_gb: int,
    threshold_pct: float,
    filters: list[str] | None = None,
    root: str | Path = "/var/lib/buildkit",
) -> bool:
    """
    Prune cache if disk usage exceeds threshold_pct (0-100).
    Returns True if a prune was attempted.
    """
    used, total = buildkit_disk_usage(root)
    if total <= 0:
        logger.warning("Skipping BuildKit prune; unable to determine disk usage.")
        return False

    usage_pct = (used / total) * 100
    logger.info(
        "BuildKit disk usage: %.2f%% (%0.2f GiB used / %0.2f GiB total)",
        usage_pct,
        used / (1 << 30),
        total / (1 << 30),
    )
    if usage_pct < threshold_pct:
        return False

    try:
        prune_buildkit_cache(keep_storage_gb=keep_storage_gb, filters=filters)
        return True
    except Exception as e:
        logger.warning("Failed to prune BuildKit cache: %s", e)
        return False
