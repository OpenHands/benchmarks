"""Unified thread-safety context for evaluation worker threads.

Consolidates all per-thread routing infrastructure (logging handlers,
stdout/stderr redirection) into a single module with one threading.local()
and one context manager.

Key design note: The SDK's RemoteConversation spawns a WebSocket daemon
thread to process events.  Visualization callbacks (Console.print) fire
on that child thread, which does NOT have its own ``_ctx.log_file``.
To capture that output we maintain a process-level registry
(``_log_file_registry``) keyed by thread ID so child threads can inherit
the log file of the worker thread that spawned them.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import contextmanager
from typing import IO, Generator

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

# Process-level registry: thread-id → log file.
# Worker threads register here so that child threads (e.g. the SDK's
# WebSocket callback thread) can look up the log file of their creator.
_log_file_registry: dict[int, IO[str]] = {}
_registry_lock = threading.Lock()

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
        except (OSError, ValueError) as e:
            # File handler failed (closed file, disk full, etc.) —
            # fall back to stderr so the message isn't silently lost.
            if sys.__stderr__:
                sys.__stderr__.write(f"LOGGING FAILURE: {e}\n{record_msg}\n")
            else:
                raise  # Don't hide logging system failures


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
            except (OSError, ValueError):
                pass  # stderr itself failed — nothing left to fall back to


# ---------------------------------------------------------------------------
# Thread-local stdout/stderr writer
# ---------------------------------------------------------------------------


class _ThreadLocalWriter:
    """A sys.stdout/sys.stderr replacement that writes to a per-thread file.

    If the current thread has set ``_ctx.log_file``, writes go there.
    Otherwise, if the current thread was spawned by a worker that registered
    a log file in ``_log_file_registry``, writes go there (this handles the
    SDK's WebSocket callback thread which fires visualization events on a
    child thread).  Finally falls through to the original stream.
    """

    def __init__(self, original: object) -> None:
        self._original = original

    def _target(self) -> object:
        # Fast path: thread-local log file (set by instance_context)
        log_file = getattr(_ctx, "log_file", None)
        if log_file is not None:
            return log_file
        # Slow path: check if the current thread's *creator* registered
        # a log file.  threading.current_thread() is cheap; the dict
        # lookup is O(1).  We check the registry without locking for
        # read performance — stale reads are harmless (worst case a
        # write goes to the original stream for one call).
        parent_id = getattr(threading.current_thread(), "_parent_thread_id", None)
        if parent_id is not None:
            parent_file = _log_file_registry.get(parent_id)
            if parent_file is not None and not parent_file.closed:
                return parent_file
        return self._original

    # --- file-like API used by print() and the logging module ---------------

    def write(self, s: str) -> int:
        target = self._target()
        try:
            return target.write(s)  # type: ignore[union-attr]
        except (ValueError, OSError) as e:
            # Target closed/broken — try original, then __stderr__ as last resort
            try:
                return self._original.write(s)  # type: ignore[union-attr]
            except Exception:
                if sys.__stderr__:
                    sys.__stderr__.write(f"STDOUT WRITE FAILURE: {e}\n")
                raise  # Don't hide I/O failures

    def flush(self) -> None:
        target = self._target()
        try:
            target.flush()  # type: ignore[union-attr]
        except (ValueError, OSError) as e:
            try:
                self._original.flush()  # type: ignore[union-attr]
            except Exception:
                if sys.__stderr__:
                    sys.__stderr__.write(f"FLUSH FAILURE: {e}\n")
                raise  # Don't hide I/O failures

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


def _patch_thread_parent_tracking() -> None:
    """Monkey-patch threading.Thread.__init__ to record the parent thread ID.

    This lets child threads (e.g. the SDK's WebSocket callback thread)
    discover which worker thread spawned them, so _ThreadLocalWriter can
    route their stdout to the correct per-instance output file.
    """
    _orig_init = threading.Thread.__init__

    def _init_with_parent(
        self: threading.Thread, *args: object, **kwargs: object
    ) -> None:
        _orig_init(self, *args, **kwargs)  # type: ignore[arg-type]
        self._parent_thread_id = threading.current_thread().ident  # type: ignore[attr-defined]

    threading.Thread.__init__ = _init_with_parent  # type: ignore[assignment]


def initialize() -> None:
    """One-time setup: install handlers on root logger, install
    _ThreadLocalWriter on sys.stdout/stderr, suppress OTel logger,
    patch Thread to track parent IDs, set main-thread defaults.

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

            # Patch Thread to record parent thread ID for log-file inheritance
            _patch_thread_parent_tracking()

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

        # Register in the process-level registry so child threads
        # (e.g. WebSocket callback thread) can inherit this log file.
        tid = threading.current_thread().ident
        if tid is not None:
            with _registry_lock:
                _log_file_registry[tid] = output_file

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
        # Unregister from process-level registry
        tid = threading.current_thread().ident
        if tid is not None:
            with _registry_lock:
                _log_file_registry.pop(tid, None)

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
