"""Thread-local context for evaluation worker threads.

Provides per-thread logging and stdout/stderr redirection for asyncio workers.

Why thread-local routing is necessary:
- Third-party code (SDK, OpenTelemetry, litellm) calls logging.getLogger()
  and writes to the root logger. We can't pass explicit loggers to them.
- sys.stdout/stderr are process-global. SDK code and tqdm print() to them.
  The only way to route per-thread is a wrapper that checks thread-local state.

This is the standard pattern (same as Django's locale.activate() per-request).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import contextmanager
from typing import Generator, TextIO

from benchmarks.utils.console_logging import (
    BG_BLUE,
    CYAN_BRIGHT,
    _ColorFormatter,
    _ConsoleFilter,
    _PlainFormatter,
    _rich_logging_enabled,
    format_line,
)


# Single thread-local for all per-thread state
_ctx = threading.local()
_init_lock = threading.Lock()
_initialized = False


# ---------------------------------------------------------------------------
# Logging handlers (one instance each, attached to root logger)
# ---------------------------------------------------------------------------


class _FileHandler(logging.Handler):
    """Writes to the per-thread file stored in _ctx.file_handler."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

    def emit(self, record: logging.LogRecord) -> None:
        fh = getattr(_ctx, "file_handler", None)
        if fh is None:
            return  # Thread has no log file configured
        msg = self.format(record)
        try:
            fh.stream.write(msg + "\n")
            fh.stream.flush()
        except (OSError, ValueError) as e:
            # Write failed — log to stderr so we know something is wrong
            if sys.__stderr__:
                sys.__stderr__.write(f"[logging failed: {e}] {msg}\n")


class _ConsoleHandler(logging.Handler):
    """Writes to stderr with per-thread formatter from _ctx."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)

    def emit(self, record: logging.LogRecord) -> None:
        fmt = getattr(_ctx, "console_formatter", None)
        if fmt is None:
            return  # Thread has no console formatter
        level = getattr(_ctx, "console_level", logging.WARNING)
        filt = getattr(_ctx, "console_filter", None)
        # Check level and filter
        if record.levelno < level:
            if not (filt and filt.filter(record)):
                return
        if sys.__stderr__:
            try:
                sys.__stderr__.write(fmt.format(record) + "\n")
                sys.__stderr__.flush()
            except (OSError, ValueError):
                pass  # stderr failed — nothing left to try


# ---------------------------------------------------------------------------
# stdout/stderr wrapper
# ---------------------------------------------------------------------------


class _StdoutWrapper:
    """Redirects writes to _ctx.output_file or original stream.

    Explicit properties instead of __getattr__ to avoid hiding bugs.
    """

    def __init__(self, original: TextIO) -> None:
        self._original = original

    def _target(self) -> TextIO:
        return getattr(_ctx, "output_file", None) or self._original

    def write(self, s: str) -> int:
        return self._target().write(s)

    def flush(self) -> None:
        self._target().flush()

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def initialize() -> None:
    """One-time setup: install routed handlers and stdout wrapper.

    Also sets up main-thread defaults so it can log without instance_context.
    """
    global _initialized
    with _init_lock:
        if _initialized:
            return
        # Install routed handlers on root logger
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(_FileHandler())
        root.addHandler(_ConsoleHandler())
        root.setLevel(logging.DEBUG)

        # Install stdout/stderr wrappers
        if not isinstance(sys.stdout, _StdoutWrapper):
            sys.stdout = _StdoutWrapper(sys.stdout)  # type: ignore[assignment]
            sys.stderr = _StdoutWrapper(sys.stderr)  # type: ignore[assignment]

        # Suppress noisy OTel context-detach errors
        logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

        _initialized = True

    # Main-thread defaults (console only, WARNING+)
    if not hasattr(_ctx, "console_formatter"):
        _ctx.console_formatter = _PlainFormatter("main")
        _ctx.console_filter = None
        _ctx.console_level = logging.WARNING


@contextmanager
def instance_context(log_dir: str, instance_id: str) -> Generator[None, None, None]:
    """Set up per-instance logging and output redirection for this thread.

    Args:
        log_dir: Directory for log files
        instance_id: The evaluation instance ID
    """
    initialize()

    log_path = os.path.join(log_dir, f"instance_{instance_id}.log")
    output_path = os.path.join(log_dir, f"instance_{instance_id}.output.log")
    short_id = (
        instance_id.split("__")[-1][:20] if "__" in instance_id else instance_id[:20]
    )
    rich_mode = _rich_logging_enabled()

    os.makedirs(log_dir, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    output_file = open(output_path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115

    # Set thread-local state
    _ctx.file_handler = logging.FileHandler(log_path)
    _ctx.output_file = output_file
    if rich_mode:
        _ctx.console_formatter = _ColorFormatter(instance_id)
        _ctx.console_filter = _ConsoleFilter()
        _ctx.console_level = logging.INFO
    else:
        _ctx.console_formatter = _PlainFormatter(instance_id)
        _ctx.console_filter = None
        _ctx.console_level = logging.WARNING

    # Print startup message
    if rich_mode:
        if sys.__stderr__:
            sys.__stderr__.write(
                format_line(
                    short_id=short_id,
                    tag="START",
                    message=f"{instance_id} | Logs: {log_path}",
                    tag_bg=BG_BLUE,
                    message_color=CYAN_BRIGHT,
                    newline_before=True,
                )
                + "\n"
            )
            sys.__stderr__.flush()
    else:
        # Temporarily allow INFO for startup message
        _ctx.console_level = logging.INFO
        logging.getLogger().info(
            f"=== Evaluation Started (instance {instance_id}) ===\n"
            f"    • tail -f {log_path}      (logger)\n"
            f"    • tail -f {output_path}   (stdout/stderr)"
        )
        _ctx.console_level = logging.WARNING

    try:
        yield
    finally:
        # Clear thread-local state
        for attr in ("file_handler", "output_file", "console_formatter", "console_filter"):
            if hasattr(_ctx, attr):
                delattr(_ctx, attr)
        log_file.close()
        output_file.close()
