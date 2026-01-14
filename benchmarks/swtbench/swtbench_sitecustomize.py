"""
Runtime-injected sitecustomize for SWT-Bench harness profiling.

This file is copied into the swt-bench checkout as sitecustomize.py to collect
coarse-grained timing events without modifying upstream code. It is activated
only when PROFILE_SWTBENCH/SWTBENCH_PROFILE_JSON are set by the caller.
"""

import atexit
import importlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


PROFILE_PATH = os.environ.get("SWTBENCH_PROFILE_JSON", "swtbench_profile.json")
_events: list[Dict[str, Any]] = []
_lock = threading.Lock()
_start_ns = time.perf_counter_ns()


def _record(name: str, extra: Optional[Dict[str, Any]] = None):
    start_ns = time.perf_counter_ns()

    def _end(status: str = "ok", extra_end: Optional[Dict[str, Any]] = None):
        end_ns = time.perf_counter_ns()
        payload: Dict[str, Any] = {
            "name": name,
            "status": status,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ms": (end_ns - start_ns) / 1_000_000,
        }
        if extra:
            payload.update(extra)
        if extra_end:
            payload.update(extra_end)
        with _lock:
            _events.append(payload)

    return _end


def _safe_patch(module, attr: str, wrapper):
    try:
        original = getattr(module, attr)
        setattr(module, attr, wrapper(original))
    except Exception:
        # If patching fails, skip silently to avoid impacting the harness.
        return


# Patch swt-bench functions if available
try:
    run_evaluation = importlib.import_module("run_evaluation")  # type: ignore[assignment]

    def _wrap_run_instances(original):
        def _inner(predictions, instances, *args, **kwargs):
            done = _record(
                "run_instances",
                {"instance_count": len(instances) if instances is not None else None},
            )
            try:
                return original(predictions, instances, *args, **kwargs)
            finally:
                done()

        return _inner

    def _wrap_run_eval_exec_spec(original):
        def _inner(exec_spec, model_patch, *args, **kwargs):
            done = _record(
                "run_eval_exec_spec",
                {"instance_id": getattr(exec_spec, "instance_id", None)},
            )
            try:
                return original(exec_spec, model_patch, *args, **kwargs)
            finally:
                done()

        return _inner

    _safe_patch(run_evaluation, "run_instances", _wrap_run_instances)
    _safe_patch(run_evaluation, "run_eval_exec_spec", _wrap_run_eval_exec_spec)
except Exception:
    pass

try:
    docker_build = importlib.import_module("src.docker_build")  # type: ignore[assignment]

    def _wrap_build_image(original):
        def _inner(image_name, *args, **kwargs):
            done = _record("docker_build", {"image_name": image_name})
            try:
                return original(image_name, *args, **kwargs)
            finally:
                done()

        return _inner

    _safe_patch(docker_build, "build_image", _wrap_build_image)
except Exception:
    pass


def _flush() -> None:
    end_ns = time.perf_counter_ns()
    payload = {
        "started_ns": _start_ns,
        "ended_ns": end_ns,
        "duration_ms": (end_ns - _start_ns) / 1_000_000,
        "events": _events,
    }
    try:
        Path(PROFILE_PATH).write_text(json.dumps(payload, indent=2))
    except Exception:
        # Avoid raising during interpreter shutdown
        return


atexit.register(_flush)
