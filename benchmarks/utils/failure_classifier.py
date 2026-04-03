"""Classify evaluation failures to decide whether resource escalation is warranted.

The retry loop in evaluation.py retries on *any* exception.  Previously every
retry also doubled the ``resource_factor`` (CPU / memory multiplier) under the
assumption that the failure was resource-related.  In practice many failures
are protocol / model / session errors where more resources won't help.

This module provides :func:`classify_failure` which inspects the exception
string and returns a :class:`FailureCategory`.  The retry loop uses the
category to decide whether to bump ``resource_factor``.

See https://github.com/OpenHands/evaluation/issues/408 for context.
"""

from __future__ import annotations

import enum
import re


class FailureCategory(enum.Enum):
    """Broad failure bucket that drives the retry / escalation strategy.

    ``RESOURCE``  – failures plausibly caused by insufficient runtime resources
                    (startup failures, OOM, image pull issues).
                    → retry **with** resource_factor escalation.

    ``NON_RESOURCE`` – failures clearly unrelated to resource pressure
                       (protocol / model / session / application errors).
                       → retry at the **same** resource_factor.
    """

    RESOURCE = "resource"
    NON_RESOURCE = "non_resource"


# ---------------------------------------------------------------------------
# Pattern table
# ---------------------------------------------------------------------------
# Each entry is ``(compiled_regex, category)``.  First match wins.
# Order matters: more specific patterns should come before generic ones.

_PATTERNS: list[tuple[re.Pattern[str], FailureCategory]] = [
    # ── Non-resource: ACP / model / protocol errors ────────────────────
    (
        re.compile(r"ACPPromptError: terminated", re.IGNORECASE),
        FailureCategory.NON_RESOURCE,
    ),
    (
        re.compile(r"ACPPromptError: Internal Server Error", re.IGNORECASE),
        FailureCategory.NON_RESOURCE,
    ),
    (
        re.compile(r"ACPPromptError: Model stream ended", re.IGNORECASE),
        FailureCategory.NON_RESOURCE,
    ),
    (re.compile(r"ACPPromptError:", re.IGNORECASE), FailureCategory.NON_RESOURCE),
    (re.compile(r"ACP prompt timed out", re.IGNORECASE), FailureCategory.NON_RESOURCE),
    (re.compile(r"ACP error: terminated", re.IGNORECASE), FailureCategory.NON_RESOURCE),
    (
        re.compile(r"ACP error: Internal Server Error", re.IGNORECASE),
        FailureCategory.NON_RESOURCE,
    ),
    (re.compile(r"ACP error:", re.IGNORECASE), FailureCategory.NON_RESOURCE),
    # ── Non-resource: HTTP / transport errors from the model layer ─────
    (
        re.compile(r"503 Service Unavailable", re.IGNORECASE),
        FailureCategory.NON_RESOURCE,
    ),
    (
        re.compile(r"Server disconnected without sending a response", re.IGNORECASE),
        FailureCategory.NON_RESOURCE,
    ),
    (
        re.compile(r"Remote conversation ended with error", re.IGNORECASE),
        FailureCategory.NON_RESOURCE,
    ),
    # ── Non-resource: malformed model output ───────────────────────────
    (
        re.compile(r"malformed function call", re.IGNORECASE),
        FailureCategory.NON_RESOURCE,
    ),
    (
        re.compile(r"temp and top_p cannot both be specified", re.IGNORECASE),
        FailureCategory.NON_RESOURCE,
    ),
    (
        re.compile(r"does not support parameters", re.IGNORECASE),
        FailureCategory.NON_RESOURCE,
    ),
    # ── Resource: image / registry problems (may resolve with rebuild) ─
    (
        re.compile(r"does not exist in container registry", re.IGNORECASE),
        FailureCategory.RESOURCE,
    ),
    (re.compile(r"ImagePullBackOff", re.IGNORECASE), FailureCategory.RESOURCE),
    (re.compile(r"ErrImagePull", re.IGNORECASE), FailureCategory.RESOURCE),
    # ── Resource: runtime startup / readiness ──────────────────────────
    (re.compile(r"Runtime not yet ready", re.IGNORECASE), FailureCategory.RESOURCE),
    (re.compile(r"OOMKill", re.IGNORECASE), FailureCategory.RESOURCE),
    (re.compile(r"OutOfMemory", re.IGNORECASE), FailureCategory.RESOURCE),
    (re.compile(r"cannot be scheduled", re.IGNORECASE), FailureCategory.RESOURCE),
    (re.compile(r"Insufficient (cpu|memory)", re.IGNORECASE), FailureCategory.RESOURCE),
]


def classify_failure(error: Exception) -> FailureCategory:
    """Classify an evaluation failure as resource-related or not.

    Inspects the full exception chain (``__cause__``, ``__context__``) so that
    wrapped errors are also considered.

    Returns :attr:`FailureCategory.RESOURCE` by default (unknown errors are
    assumed to be possibly resource-related so that the existing escalation
    behaviour is preserved for novel failure modes).
    """
    messages: list[str] = []
    exc: BaseException | None = error
    while exc is not None:
        messages.append(str(exc))
        # Walk the exception chain
        exc = exc.__cause__ or exc.__context__

    combined = " | ".join(messages)

    for pattern, category in _PATTERNS:
        if pattern.search(combined):
            return category

    # Default: treat unknown failures as potentially resource-related so we
    # don't silently suppress escalation for genuinely new failure modes.
    return FailureCategory.RESOURCE
