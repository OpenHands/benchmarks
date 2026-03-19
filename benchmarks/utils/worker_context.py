"""Simple instance-tagged logging for evaluation workers.

Each thread sets its current instance ID. A logging filter automatically
prefixes all log messages with [instance_id]. After evaluation, logs can
be split into per-instance files.

This replaces complex thread-local routing with a simple prefix + post-process approach.
"""

import logging
import os
import re
import sys
import threading
from contextlib import contextmanager
from typing import IO, Generator, TextIO


# Thread-local storage for current instance ID (just a string, not complex objects)
_current_instance: threading.local = threading.local()

# Shared output file for stdout/stderr capture
_output_file: IO[str] | None = None
_output_lock = threading.Lock()


class _InstanceFilter(logging.Filter):
    """Adds [instance_id] prefix to all log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        instance_id = getattr(_current_instance, "id", None)
        if instance_id:
            record.msg = f"[{instance_id}] {record.msg}"
        return True


class _OutputCapture:
    """Captures stdout/stderr to shared file with instance prefix."""

    def __init__(self, original: TextIO) -> None:
        self._original = original

    def write(self, s: str) -> int:
        # Always write to original (terminal)
        result = self._original.write(s)

        # Also write to shared output file with instance prefix
        if _output_file and s.strip():
            instance_id = getattr(_current_instance, "id", None)
            if instance_id:
                with _output_lock:
                    _output_file.write(f"[{instance_id}] {s}")
                    if not s.endswith("\n"):
                        _output_file.write("\n")
                    _output_file.flush()
        return result

    def flush(self) -> None:
        self._original.flush()

    @property
    def encoding(self) -> str:
        return self._original.encoding

    @property
    def closed(self) -> bool:
        return self._original.closed

    def isatty(self) -> bool:
        return self._original.isatty()

    def fileno(self) -> int:
        return self._original.fileno()


# Global state
_filter: _InstanceFilter | None = None
_initialized = False


def initialize(log_dir: str) -> None:
    """Set up logging and stdout capture to shared files.

    Call once at the start of evaluation. All logs go to evaluation.log,
    stdout/stderr go to evaluation.output.log, both with [instance_id] prefix.
    """
    global _filter, _output_file, _initialized

    if _initialized:
        return

    os.makedirs(log_dir, exist_ok=True)

    # Set up logging
    log_path = os.path.join(log_dir, "evaluation.log")
    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.FileHandler(log_path, mode="a")
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    handler.setLevel(logging.INFO)
    root.addHandler(handler)

    # Also log to stderr for visibility
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    console.setLevel(logging.WARNING)
    root.addHandler(console)

    root.setLevel(logging.DEBUG)

    # Add filter to prefix instance ID
    _filter = _InstanceFilter()
    root.addFilter(_filter)

    # Set up stdout/stderr capture
    output_path = os.path.join(log_dir, "evaluation.output.log")
    _output_file = open(output_path, "a", buffering=1, encoding="utf-8")

    if not isinstance(sys.stdout, _OutputCapture):
        sys.stdout = _OutputCapture(sys.stdout)  # type: ignore[assignment]
        sys.stderr = _OutputCapture(sys.stderr)  # type: ignore[assignment]

    # Suppress noisy OpenTelemetry context errors
    logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

    _initialized = True


@contextmanager
def instance_context(log_dir: str, instance_id: str) -> Generator[None, None, None]:
    """Context manager that sets the current instance ID for logging.

    All log messages and stdout/stderr within this context will be
    prefixed with [instance_id].
    """
    # Ensure logging is initialized
    if not _initialized:
        initialize(log_dir)

    _current_instance.id = instance_id
    try:
        logging.getLogger().info("start")
        yield
    finally:
        _current_instance.id = None


def split_logs(log_dir: str) -> None:
    """Split shared log files into per-instance files.

    Call after evaluation completes. Creates:
    - logs/instance_{id}.log (from evaluation.log)
    - logs/instance_{id}.output.log (from evaluation.output.log)
    """
    _split_file(
        os.path.join(log_dir, "evaluation.log"),
        log_dir,
        "instance_{}.log",
    )
    _split_file(
        os.path.join(log_dir, "evaluation.output.log"),
        log_dir,
        "instance_{}.output.log",
    )


def _split_file(shared_path: str, output_dir: str, filename_template: str) -> None:
    """Split a shared log file into per-instance files."""
    if not os.path.exists(shared_path):
        return

    files: dict[str, IO[str]] = {}
    pattern = re.compile(r"\[([^\]]+)\]")

    try:
        with open(shared_path, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    inst_id = match.group(1)
                    if inst_id not in files:
                        path = os.path.join(output_dir, filename_template.format(inst_id))
                        files[inst_id] = open(path, "w", encoding="utf-8")
                    files[inst_id].write(line)
    finally:
        for fh in files.values():
            fh.close()

    if files:
        logging.getLogger().info(
            "Split %s into %d per-instance files", shared_path, len(files)
        )
