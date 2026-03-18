"""Unified thread-safety context for evaluation worker threads.

Consolidates all per-thread routing infrastructure (logging handlers,
stdout/stderr redirection) into a single module with one threading.local()
and one context manager.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import contextmanager
from typing import Generator

from benchmarks.utils.console_logging import (
    BG_BLUE,
    CYAN_BRIGHT,
    _ColorFormatter,
    _ConsoleFilter,
    _PlainFormatter,
    _rich_logging_enabled,
    format_line,
)


# ---------------------------------------------------------------------------
# Single thread-local for ALL per-thread state
# ---------------------------------------------------------------------------
_ctx = threading.local()

# One-time initialization guard
_setup_lock = threading.Lock()
_initialized = False


# ---------------------------------------------------------------------------
# Thread-routed logging handlers
# ---------------------------------------------------------------------------


class _RoutedFileHandler(logging.Handler):
    """Routes log records to per-thread file handlers via _ctx.

    A single instance is attached to the root logger. Each worker thread
    stores its own FileHandler in _ctx.file_handler, and this handler
    delegates to whichever FileHandler the current thread has.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

    def emit(self, record: logging.LogRecord) -> None:
        fh: logging.FileHandler | None = getattr(_ctx, "file_handler", None)
        if fh is None:
            return
        record_msg = self.format(record)
        try:
            fh.stream.write(record_msg + "\n")
            fh.stream.flush()
        except Exception:
            pass


class _RoutedConsoleHandler(logging.Handler):
    """Routes console output with per-thread formatter via _ctx.

    All output goes to sys.__stderr__ to protect stdout (used for JSON
    output parsing by shell scripts). Each worker thread stores its own
    formatter/filter in _ctx.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)

    def emit(self, record: logging.LogRecord) -> None:
        fmt = getattr(_ctx, "console_formatter", None)
        filt = getattr(_ctx, "console_filter", None)
        level = getattr(_ctx, "console_level", logging.WARNING)
        if record.levelno < level:
            if filt and not filt.filter(record):
                return
            elif not filt:
                return
        if fmt is None:
            return
        stream = sys.__stderr__
        if stream:
            try:
                msg = fmt.format(record)
                stream.write(msg + "\n")
                stream.flush()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Thread-local stdout/stderr writer
# ---------------------------------------------------------------------------


class _ThreadLocalWriter:
    """A sys.stdout/sys.stderr replacement that writes to a per-thread file.

    If the current thread has set ``_ctx.log_file``, writes go there.
    Otherwise writes fall through to the original stream (usually the real
    terminal stdout/stderr).
    """

    def __init__(self, original: object) -> None:
        self._original = original

    def _target(self) -> object:
        return getattr(_ctx, "log_file", None) or self._original

    # --- file-like API used by print() and the logging module ---------------

    def write(self, s: str) -> int:
        target = self._target()
        try:
            return target.write(s)  # type: ignore[union-attr]
        except ValueError:
            # Handle "I/O operation on closed file" gracefully –
            # fall back to original stream instead of crashing.
            return self._original.write(s)  # type: ignore[union-attr]

    def flush(self) -> None:
        target = self._target()
        try:
            target.flush()  # type: ignore[union-attr]
        except ValueError:
            self._original.flush()  # type: ignore[union-attr]

    @property
    def encoding(self) -> str:
        return self._target().encoding  # type: ignore[union-attr]

    @property
    def closed(self) -> bool:
        return self._target().closed  # type: ignore[union-attr]

    def isatty(self) -> bool:
        return self._original.isatty()  # type: ignore[union-attr]

    def fileno(self) -> int:
        return self._original.fileno()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def initialize() -> None:
    """One-time setup: install handlers on root logger, install
    _ThreadLocalWriter on sys.stdout/stderr, suppress OTel logger,
    set main-thread defaults.

    Idempotent — safe to call multiple times.
    """
    global _initialized
    with _setup_lock:
        if not _initialized:
            # Replace root logger handlers with thread-routed handlers
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            root_logger.addHandler(_RoutedFileHandler())
            root_logger.addHandler(_RoutedConsoleHandler())
            root_logger.setLevel(logging.DEBUG)

            # Install thread-local writers for stdout/stderr
            if not isinstance(sys.stdout, _ThreadLocalWriter):
                sys.stdout = _ThreadLocalWriter(sys.stdout)  # type: ignore[assignment]
                sys.stderr = _ThreadLocalWriter(sys.stderr)  # type: ignore[assignment]

            # Suppress noisy OpenTelemetry context-detach errors that happen
            # when spans created in the main thread are ended in worker threads.
            logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

            _initialized = True

    # Set main-thread defaults (plain formatter, WARNING+ only)
    if not hasattr(_ctx, "console_formatter"):
        _ctx.console_formatter = _PlainFormatter("main")
        _ctx.console_filter = None
        _ctx.console_level = logging.WARNING


@contextmanager
def instance_context(log_dir: str, instance_id: str) -> Generator[None, None, None]:
    """Single context manager replacing setup_instance_logging() and
    redirect_stdout_stderr().

    Sets up:
    1. Thread-routed logging (file handler + console formatter/filter)
    2. stdout/stderr redirection to per-instance output log file

    All state is stored in ``_ctx`` and restored on exit.

    Args:
        log_dir: Directory for log files
        instance_id: The evaluation instance ID
    """
    # Ensure global handlers are installed (idempotent)
    initialize()

    log_file_path = os.path.join(log_dir, f"instance_{instance_id}.log")
    output_log_path = os.path.join(log_dir, f"instance_{instance_id}.output.log")
    short_id = (
        instance_id.split("__")[-1][:20] if "__" in instance_id else instance_id[:20]
    )
    rich_mode = _rich_logging_enabled()

    # Save previous state for restoration
    prev_file_handler: logging.FileHandler | None = getattr(_ctx, "file_handler", None)
    prev_console_formatter = getattr(_ctx, "console_formatter", None)
    prev_console_filter = getattr(_ctx, "console_filter", None)
    prev_console_level = getattr(_ctx, "console_level", None)
    had_log_file = hasattr(_ctx, "log_file")
    prev_log_file = getattr(_ctx, "log_file", None)

    output_file = None
    fh = None
    try:
        os.makedirs(log_dir, exist_ok=True)

        # --- Set up logging file handler ---
        if prev_file_handler is not None:
            try:
                prev_file_handler.close()
            except Exception:
                pass

        fh = logging.FileHandler(log_file_path)
        _ctx.file_handler = fh

        # --- Set up console formatter/filter ---
        if rich_mode:
            _ctx.console_formatter = _ColorFormatter(instance_id)
            _ctx.console_filter = _ConsoleFilter()
            _ctx.console_level = logging.INFO
        else:
            _ctx.console_formatter = _PlainFormatter(instance_id)
            _ctx.console_filter = None
            _ctx.console_level = logging.WARNING

        # --- Set up stdout/stderr redirect ---
        output_file = open(  # noqa: SIM115
            output_log_path, "a", buffering=1, encoding="utf-8"
        )
        _ctx.log_file = output_file

        # --- Print startup message ---
        root_logger = logging.getLogger()
        if rich_mode:
            print(
                format_line(
                    short_id=short_id,
                    tag="START",
                    message=f"{instance_id} | Logs: {log_file_path}",
                    tag_bg=BG_BLUE,
                    message_color=CYAN_BRIGHT,
                    newline_before=True,
                ),
                file=sys.__stderr__,
            )
            if sys.__stderr__ is not None:
                sys.__stderr__.flush()
        else:
            # Temporarily allow INFO for the startup message
            _ctx.console_level = logging.INFO
            root_logger.info(
                f"""
    === Evaluation Started (instance {instance_id}) ===
    View live output:
    • tail -f {log_file_path}          (logger)
    • tail -f {output_log_path}   (stdout/stderr)
    ===============================================
    """.strip()
            )
            # Restore WARNING+ for console after startup message
            _ctx.console_level = logging.WARNING

        yield

    finally:
        # Restore previous state
        if had_log_file:
            _ctx.log_file = prev_log_file
        elif hasattr(_ctx, "log_file"):
            del _ctx.log_file

        if prev_console_formatter is not None:
            _ctx.console_formatter = prev_console_formatter
        if prev_console_filter is not None:
            _ctx.console_filter = prev_console_filter
        if prev_console_level is not None:
            _ctx.console_level = prev_console_level

        # Don't restore prev_file_handler (it was closed above);
        # just clear if there was no previous one
        if prev_file_handler is None and hasattr(_ctx, "file_handler"):
            del _ctx.file_handler

        # Close files
        if output_file is not None and not output_file.closed:
            output_file.close()
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
