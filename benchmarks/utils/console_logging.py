from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from openhands.sdk import Event

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
RICH_LOGGING_ENV_VAR = "RICH_LOGGING"


def _rich_logging_enabled() -> bool:
    """Check if rich logging is enabled via environment variable."""
    return os.getenv(RICH_LOGGING_ENV_VAR, "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# ANSI color constants
# ---------------------------------------------------------------------------
RESET = "\033[0m"
DIM = "\033[2m"

BLACK = "\033[30m"
WHITE = "\033[37m"
RED = "\033[31m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
CYAN_BRIGHT = "\033[96m"
WHITE_BOLD = "\033[1;37m"

BG_BLACK = "\033[40m"
BG_RED = "\033[41m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"
BG_MAGENTA = "\033[45m"
BG_CYAN = "\033[46m"
BG_WHITE = "\033[47m"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_WS_RE = re.compile(r"\s+")


def strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences."""
    return _ANSI_RE.sub("", s)


# ---------------------------------------------------------------------------
# Line formatting helpers
# ---------------------------------------------------------------------------


def _format_timestamp(dt: datetime | None = None) -> str:
    dt = dt or datetime.now()
    return f"{DIM}{dt.strftime('%Y-%m-%d %H:%M:%S')}{RESET}"


def _format_sample_id(short_id: str) -> str:
    return f"{DIM}[{short_id}]{RESET}"


def _format_tag(tag: str, *, bg: str, fg: str = WHITE_BOLD) -> str:
    return f"{bg}{fg} {tag} {RESET}"


def format_line(
    *,
    short_id: str,
    tag: str,
    message: str,
    tag_bg: str,
    tag_fg: str = WHITE_BOLD,
    message_color: str = CYAN_BRIGHT,
    newline_before: bool = False,
    newline_after: bool = False,
) -> str:
    line = (
        f"{_format_timestamp()} {_format_sample_id(short_id)} "
        f"{_format_tag(tag, bg=tag_bg, fg=tag_fg)} {message_color}{message}{RESET}"
    )
    if newline_before:
        line = "\n" + line
    if newline_after:
        line = line + "\n"
    return line


def _format_box_prefix(*, short_id: str, tag: str, tag_bg: str) -> str:
    return (
        f"{_format_timestamp()} {_format_sample_id(short_id)} "
        f"{_format_tag(tag, bg=tag_bg)}     │"
    )


# ---------------------------------------------------------------------------
# Trajectory line formatting (per-event console output)
# ---------------------------------------------------------------------------


def _one_line(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _extract_tool_input_preview(event: "Event") -> str:
    action = getattr(event, "action", None)
    if action is None:
        return ""
    for attr, label in (
        ("command", "cmd"),
        ("path", "path"),
        ("file_path", "path"),
        ("target_file", "path"),
        ("query", "query"),
    ):
        if hasattr(action, attr):
            v = getattr(action, attr)
            if isinstance(v, str) and v.strip():
                return f" {label}={_truncate(_one_line(v), 120)!r}"
    for attr, label in (
        ("new_string", "new"),
        ("new_str", "new"),
        ("old_string", "old"),
    ):
        if hasattr(action, attr):
            v = getattr(action, attr)
            if isinstance(v, str) and v.strip():
                return f" {label}={_truncate(_one_line(v), 80)!r}"
    return ""


def format_trajectory_line(
    event: "Event", *, short_id: str, tool_call_index: int
) -> str | None:
    """Format a single-line trajectory summary"""
    # Avoid top-level import to prevent circular dependencies
    from openhands.sdk.event import (
        ActionEvent,
        AgentErrorEvent,
        MessageEvent,
        ObservationEvent,
    )

    # -------------------------------------------------------------------------
    # Tool call logs (ActionEvent with tool_name)
    # #N = tool call counter within this instance (1st call, 2nd call, ...)
    # Example log:
    #   10:30:45 [django-12345]  TOOL  │ ▶ bash #1 cmd='ls'
    # -------------------------------------------------------------------------
    if isinstance(event, ActionEvent) and event.tool_name:
        thought = ""
        if event.action is not None and getattr(event.action, "thought", None):
            thought = str(getattr(event.action, "thought") or "")[:50]
            if thought:
                thought = f" {DIM}// {thought}...{RESET}"
        prefix = _format_box_prefix(short_id=short_id, tag="TOOL", tag_bg=BG_BLUE)
        args = _extract_tool_input_preview(event)
        args_dim = f"{DIM}{args}{RESET}" if args else ""
        return (
            f"{prefix} {WHITE}▶ {event.tool_name} #{tool_call_index}{RESET}"
            f"{args_dim}{thought}"
        )

    # -------------------------------------------------------------------------
    # Tool result log (ObservationEvent)
    # Example log:
    #   10:30:46 [django-12345]  TOOL   │   └─ ok           (exit_code=0)
    #   10:30:47 [django-12345]  WARN   │   └─ exit=1       (exit_code!=0)
    #   10:30:48 [django-12345]  WARN   │   └─ tool_error   (is_error=True)
    # -------------------------------------------------------------------------
    if isinstance(event, ObservationEvent):
        obs = event.observation
        exit_code = getattr(obs, "exit_code", None)
        is_error = bool(getattr(obs, "is_error", False))
        if exit_code is not None:
            if exit_code == 0:
                prefix = _format_box_prefix(
                    short_id=short_id, tag="TOOL", tag_bg=BG_BLUE
                )
                return f"{prefix}   {WHITE}└─ ok{RESET}"
            prefix = _format_box_prefix(short_id=short_id, tag="WARN", tag_bg=BG_YELLOW)
            return f"{prefix}   {YELLOW}└─ exit={exit_code}{RESET}"
        if is_error:
            prefix = _format_box_prefix(short_id=short_id, tag="WARN", tag_bg=BG_YELLOW)
            return f"{prefix}   {YELLOW}└─ tool_error{RESET}"
        return None  # Skip non-terminal observations (e.g., intermediate file reads)

    # -------------------------------------------------------------------------
    # Agent-side error log (AgentErrorEvent)
    # Example output:
    #   10:30:49 [django-12345]  ERROR  │   └─ error
    # -------------------------------------------------------------------------
    if isinstance(event, AgentErrorEvent):
        prefix = _format_box_prefix(short_id=short_id, tag="ERROR", tag_bg=BG_RED)
        return f"{prefix}   {RED}└─ error{RESET}"

    # -------------------------------------------------------------------------
    # Agent text message (MessageEvent from agent)
    # Displays the message content or the full message object if content is empty
    # Example output:
    #   10:30:50 [django-12345]  MESSAGE│ I found the bug in line 42...
    #   10:30:51 [django-12345]  MESSAGE│ (empty) {"tool_calls": [...]}
    # -------------------------------------------------------------------------
    if isinstance(event, MessageEvent) and event.source == "agent":
        llm_msg = event.llm_message
        prefix = _format_box_prefix(short_id=short_id, tag="MESSAGE", tag_bg=BG_MAGENTA)
        if llm_msg and hasattr(llm_msg, "content") and llm_msg.content:
            for block in llm_msg.content:
                if hasattr(block, "text"):
                    text = str(getattr(block, "text") or "")[:60].replace("\n", " ")
                    if text:
                        return f"{prefix}{WHITE_BOLD}{text}...{RESET}"
        try:
            payload = (
                _one_line(llm_msg.model_dump_json(exclude_none=True))
                if llm_msg
                else "null"
            )
        except Exception:
            payload = "null"
        return f"{prefix}{WHITE_BOLD}(empty) {payload}{RESET}"

    # Other events (user messages, system events, etc.) are not displayed
    return None


def print_trajectory_line(
    event: "Event", *, short_id: str, tool_call_index: int
) -> None:
    if not _rich_logging_enabled():
        return
    try:
        line = format_trajectory_line(
            event, short_id=short_id, tool_call_index=tool_call_index
        )
        if line and sys.__stdout__ is not None:
            print(line, file=sys.__stdout__)
            sys.__stdout__.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Logger setup for multiprocessing workers
# ---------------------------------------------------------------------------


class _ConsoleFilter(logging.Filter):
    """Only pass important messages to console."""

    IMPORTANT_PATTERNS = (
        "[INSTANCE]",
        "[LLM_ANALYSIS]",
        "[EMPTY_PATCH]",
        "[PATCH_DIAG]",
        "=== Evaluation",
        "Docker workspace",
        "repo_path:",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        msg = record.getMessage()
        return any(p in msg for p in self.IMPORTANT_PATTERNS)


class _ColorFormatter(logging.Formatter):
    """Formatter that applies colored tags based on message content/level."""

    def __init__(self, instance_id: str) -> None:
        super().__init__()
        self._short_id = (
            instance_id.split("__")[-1][:20]
            if "__" in instance_id
            else instance_id[:20]
        )

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if "[INSTANCE]" in msg:
            pretty = msg
            if "status=ERROR" in pretty:
                pretty = pretty.replace("status=ERROR", f"status={RED}ERROR{RESET}")
            return format_line(
                short_id=self._short_id,
                tag="RESULT",
                message=pretty,
                tag_bg=BG_BLUE,
                message_color="",
                newline_before=True,
                newline_after=True,
            )
        if "[LLM_ANALYSIS]" in msg:
            return format_line(
                short_id=self._short_id,
                tag="LLM",
                message=msg,
                tag_bg=BG_BLUE,
                message_color=CYAN_BRIGHT,
            )
        if "[EMPTY_PATCH]" in msg or "[PATCH_DIAG]" in msg:
            return format_line(
                short_id=self._short_id,
                tag="PATCH",
                message=msg,
                tag_bg=BG_MAGENTA,
                message_color=MAGENTA,
            )
        if record.levelno >= logging.ERROR:
            return format_line(
                short_id=self._short_id,
                tag="ERROR",
                message=msg,
                tag_bg=BG_RED,
                message_color=RED,
            )
        if record.levelno >= logging.WARNING:
            return format_line(
                short_id=self._short_id,
                tag="WARN",
                message=msg,
                tag_bg=BG_YELLOW,
                message_color=YELLOW,
            )
        return format_line(
            short_id=self._short_id,
            tag="INFO",
            message=msg,
            tag_bg=BG_WHITE,
            tag_fg=BLACK,
            message_color=WHITE,
        )


class _PlainFormatter(logging.Formatter):
    """Original formatter"""

    def __init__(self, instance_id: str) -> None:
        super().__init__(
            fmt=f"Instance {instance_id} - %(asctime)s - %(levelname)s - %(message)s"
        )


def setup_instance_logging(log_dir: str, instance_id: str) -> None:
    """
    Configure logging for a worker process handling a single instance.

    Behavior depends on RICH_LOGGING env var:
    - RICH_LOGGING=1: Colored console output with filtered messages
    - RICH_LOGGING not set: Original plain logging style

    In both modes:
    - All INFO+ logs go to per-instance file (instance_<id>.log)
    - Console shows WARNING+ and selected important patterns
    """
    log_file = os.path.join(log_dir, f"instance_{instance_id}.log")
    output_log_file = os.path.join(log_dir, f"instance_{instance_id}.output.log")
    short_id = (
        instance_id.split("__")[-1][:20] if "__" in instance_id else instance_id[:20]
    )

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    rich_mode = _rich_logging_enabled()

    # Console handler
    console_handler = logging.StreamHandler(sys.__stdout__ if rich_mode else None)
    console_handler.setLevel(logging.INFO)

    if rich_mode:
        console_handler.addFilter(_ConsoleFilter())
        console_handler.setFormatter(_ColorFormatter(instance_id))
    else:
        console_handler.setFormatter(_PlainFormatter(instance_id))

    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)

    # Print startup message
    if rich_mode:
        print(
            format_line(
                short_id=short_id,
                tag="START",
                message=f"{instance_id} | Logs: {log_file}",
                tag_bg=BG_BLUE,
                message_color=CYAN_BRIGHT,
                newline_before=True,
            ),
            file=sys.__stdout__,
        )
        if sys.__stdout__ is not None:
            sys.__stdout__.flush()
    else:
        # Original startup message style
        root_logger.info(
            f"""
    === Evaluation Started (instance {instance_id}) ===
    View live output:
    • tail -f {log_file}          (logger)
    • tail -f {output_log_file}   (stdout/stderr)
    ===============================================
    """.strip()
        )
        # Original behavior: set console to WARNING+ after initial message
        console_handler.setLevel(logging.WARNING)

    # File handler for full logs
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Instance summary (printed at end of each instance)
# ---------------------------------------------------------------------------


def _ansi(enabled: bool, code: str) -> str:
    return f"\033[{code}m" if enabled else ""


def summarize_instance(
    *,
    instance_id: str,
    conversation: Any,
    git_patch: str | None = None,
    commit_exit_code: int = 0,
    repo_has_changes: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Log a summary line for a completed instance"""
    # Lazy imports to avoid circular dependencies
    from openhands.sdk.conversation.state import ConversationExecutionStatus
    from openhands.sdk.event import ActionEvent, AgentErrorEvent, MessageEvent
    from openhands.sdk.event.conversation_error import ConversationErrorEvent
    from openhands.sdk.tool.builtins.finish import FinishAction

    if logger is None:
        logger = logging.getLogger(__name__)

    # Extract events from conversation state
    try:
        events = list(conversation.state.events)  # type: ignore[attr-defined]
    except Exception:
        events = []

    n_tool_calls = sum(
        isinstance(e, ActionEvent)
        and getattr(e, "source", None) == "agent"
        and bool(getattr(e, "tool_name", None))
        and getattr(e, "action", None) is not None
        for e in events
    )
    n_agent_msgs = sum(
        isinstance(e, MessageEvent) and getattr(e, "source", None) == "agent"
        for e in events
    )
    n_user_msgs = sum(
        isinstance(e, MessageEvent) and getattr(e, "source", None) == "user"
        for e in events
    )

    # Error counts
    n_agent_errors = sum(isinstance(e, AgentErrorEvent) for e in events)
    n_conv_errors = sum(isinstance(e, ConversationErrorEvent) for e in events)

    # Execution status
    try:
        status = conversation.state.execution_status
    except Exception:
        status = None

    healthy = (
        (n_agent_errors == 0)
        and (n_conv_errors == 0)
        and (status != ConversationExecutionStatus.ERROR if status else True)
    )

    # Check if agent used finish tool
    finished_with_finish = any(
        isinstance(e, ActionEvent)
        and isinstance(getattr(e, "action", None), FinishAction)
        for e in events
    )

    if status and status != ConversationExecutionStatus.FINISHED:
        finish_reason = f"status={status.value}"
    elif finished_with_finish:
        finish_reason = "finish_tool"
    else:
        finish_reason = "finished_no_finish_tool"

    # Patch preview
    patch_preview = (git_patch or "").strip().replace("\n", "\\n")
    if len(patch_preview) > 180:
        patch_preview = patch_preview[:180] + "…"
    patch_empty = not bool((git_patch or "").strip())

    # Colors (respect NO_COLOR env var)
    color = os.getenv("NO_COLOR") is None
    ok_c = _ansi(color, "32")
    warn_c = _ansi(color, "33")
    err_c = _ansi(color, "31")
    white_c = _ansi(color, "37")
    dim_c = _ansi(color, "2")
    reset = _ansi(color, "0")

    # Build colored tags
    health_tag = f"{ok_c}OK{reset}" if healthy else f"{warn_c}WITH_ISSUES{reset}"
    patch_tag = f"{warn_c}EMPTY{reset}" if patch_empty else f"{ok_c}NONEMPTY{reset}"
    reason_tag = (
        f"{ok_c}{finish_reason}{reset}"
        if finish_reason == "finish_tool"
        else f"{warn_c}{finish_reason}{reset}"
    )
    commit_tag = (
        f"{ok_c}{commit_exit_code}{reset}"
        if commit_exit_code == 0
        else f"{warn_c}{commit_exit_code}{reset}"
    )

    # Tool call count coloring
    if n_tool_calls == 0:
        tool_calls_tag = f"{err_c}{n_tool_calls}{reset}"
    elif n_tool_calls < n_agent_msgs:
        tool_calls_tag = f"{warn_c}{n_tool_calls}{reset}"
    else:
        tool_calls_tag = f"{white_c}{n_tool_calls}{reset}"

    # Errors coloring
    errors_tag_color = warn_c if (n_agent_errors > 0 or n_conv_errors > 0) else ok_c
    errors_tag = f"{errors_tag_color}{n_agent_errors}/{n_conv_errors}{reset}"

    if git_patch is not None:
        # Prefix with [INSTANCE] so rich console logging:
        # - passes the _ConsoleFilter (important pattern)
        # - renders with the "RESULT" tag + surrounding newlines in _ColorFormatter
        logger.info(
            "[INSTANCE] %s patch=%s commit=%s changes=%s "
            "msgs(a/u)=%d/%d tool_calls=%s "
            "errors(agent/conv)=%s end=%s %spreview='%s'%s",
            health_tag,
            patch_tag,
            commit_tag,
            "Y" if repo_has_changes else "N",
            n_agent_msgs,
            n_user_msgs,
            tool_calls_tag,
            errors_tag,
            reason_tag,
            dim_c,
            patch_preview,
            reset,
        )
    else:
        logger.info(
            "[INSTANCE] %s msgs(a/u)=%d/%d tool_calls=%s errors(agent/conv)=%s end=%s",
            health_tag,
            n_agent_msgs,
            n_user_msgs,
            tool_calls_tag,
            errors_tag,
            reason_tag,
        )
