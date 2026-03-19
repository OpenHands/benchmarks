"""Simple instance-tagged logging for evaluation workers.

Each thread sets its current instance ID. A logging filter automatically
prefixes all log messages with [instance_id]. After evaluation, logs can
be split into per-instance files.

This replaces complex thread-local routing with a simple prefix + post-process approach.
"""

import logging
import os
import re
import threading
from contextlib import contextmanager
from typing import Generator


# Thread-local storage for current instance ID (just a string, not complex objects)
_current_instance: threading.local = threading.local()


class _InstanceFilter(logging.Filter):
    """Adds [instance_id] prefix to all log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        instance_id = getattr(_current_instance, "id", None)
        if instance_id:
            record.msg = f"[{instance_id}] {record.msg}"
        return True


# Global filter instance (added to root logger once)
_filter: _InstanceFilter | None = None


def initialize(log_dir: str) -> None:
    """Set up logging to a shared file with instance prefixing.

    Call once at the start of evaluation. All logs go to evaluation.log,
    with each line prefixed by [instance_id].
    """
    global _filter

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "evaluation.log")

    root = logging.getLogger()

    # Remove existing handlers, add our shared file handler
    root.handlers.clear()

    handler = logging.FileHandler(log_path, mode="a")
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    handler.setLevel(logging.INFO)
    root.addHandler(handler)

    # Also log to stderr for visibility
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    console.setLevel(logging.WARNING)
    root.addHandler(console)

    root.setLevel(logging.DEBUG)

    # Add filter to prefix instance ID
    if _filter is None:
        _filter = _InstanceFilter()
        root.addFilter(_filter)

    # Suppress noisy OpenTelemetry context errors
    logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)


@contextmanager
def instance_context(log_dir: str, instance_id: str) -> Generator[None, None, None]:
    """Context manager that sets the current instance ID for logging.

    All log messages within this context will be prefixed with [instance_id].
    """
    # Ensure logging is initialized
    if _filter is None:
        initialize(log_dir)

    _current_instance.id = instance_id
    try:
        logging.getLogger().info("start")
        yield
    finally:
        _current_instance.id = None


def split_logs(log_dir: str) -> None:
    """Split the shared evaluation.log into per-instance files.

    Call after evaluation completes. Creates logs/instance_{id}.log files.
    """
    shared_log = os.path.join(log_dir, "evaluation.log")
    if not os.path.exists(shared_log):
        return

    files: dict[str, object] = {}
    pattern = re.compile(r"\[([^\]]+)\]")

    try:
        with open(shared_log, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    inst_id = match.group(1)
                    if inst_id not in files:
                        path = os.path.join(log_dir, f"instance_{inst_id}.log")
                        files[inst_id] = open(path, "w", encoding="utf-8")
                    files[inst_id].write(line)  # type: ignore[union-attr]
    finally:
        for fh in files.values():
            fh.close()  # type: ignore[union-attr]

    logging.getLogger().info(
        "Split evaluation.log into %d per-instance files", len(files)
    )
