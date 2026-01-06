import logging


logger = logging.getLogger(__name__)


class EvalException(Exception):
    pass


class EvalTimeoutException(Exception):
    pass


# Fatal runtime error types that indicate the runtime crashed or disconnected
# These are based on the OpenHands evaluation patterns
FATAL_RUNTIME_ERRORS = [
    "RuntimeTimeoutError",
    "RuntimeUnavailableError",
    "RuntimeDisconnectedError",
    "RuntimeNotFoundError",
    "AgentRuntimeTimeoutError",
    "AgentRuntimeUnavailableError",
    "AgentRuntimeDisconnectedError",
    "AgentRuntimeNotFoundError",
    "ConnectionError",
    "RuntimeError",  # Generic runtime errors
    "httpx.ConnectError",
    "httpx.ReadTimeout",
    "httpx.ConnectTimeout",
]


def is_fatal_runtime_error(error: str | Exception | None) -> bool:
    """Check if an error indicates a fatal runtime failure.

    Fatal runtime errors are those that indicate the runtime crashed,
    disconnected, or became unavailable. These errors should trigger
    a retry with increased resources.

    Args:
        error: The error string, exception, or None

    Returns:
        True if this is a fatal runtime error that should trigger resource increase
    """
    if error is None:
        return False

    error_str = str(error) if isinstance(error, Exception) else error

    # Also check the exception type name
    if isinstance(error, Exception):
        error_str = f"{type(error).__name__}: {error_str}"

    for fatal_error in FATAL_RUNTIME_ERRORS:
        if fatal_error in error_str:
            logger.warning(f"Fatal runtime error detected: {error_str}")
            return True

    return False
