#!/usr/bin/env python3
"""
Buildx/BuildKit utilities for image build resets and pruning.
"""

import json
import os
import re
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
