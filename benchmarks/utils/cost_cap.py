"""Per-instance cost cap callback for benchmarks.

Some evaluations have shown that a small minority of instances can consume
disproportionately large amounts of money (e.g. for the Gemini-3.5-Flash
SWE-bench run, 22 instances cost >$10 each and accounted for ~20% of the
total $1900+ spend). The dominant mechanism is a combination of:

* large iteration counts (some instances ran 300+ iterations);
* the LLM-summarising condenser, which periodically rewrites the prompt
  prefix and therefore invalidates the provider's prompt cache (cache-read
  ratio dropped from ~45% on cheap instances to ~10% on the expensive
  ones); and
* high reasoning effort, which makes every uncached call more costly.

This module provides a small, optional defence-in-depth: a
:class:`Conversation` callback that pauses the conversation once the
accumulated per-instance cost exceeds a configured threshold. It does not
attempt to fix the root cause (which would require SDK-level changes to the
condenser or to enforce ``Metrics.max_budget_per_task``); it simply caps
the blast radius.

When the cap is hit, the callback calls ``conversation.pause()`` which
takes effect at the next iteration boundary, mirroring the existing
behaviour of ``max_iteration_per_run``. Any patch produced up to that point
is still collected and submitted.
"""

from __future__ import annotations

from typing import Callable

from openhands.sdk import Event, get_logger


logger = get_logger(__name__)

ConversationCallback = Callable[[Event], None]


class CostCapCallback:
    """Callback that pauses a conversation once accumulated cost exceeds a cap.

    Use :meth:`bind` to attach the callback to the conversation after the
    conversation has been constructed; the callback can be passed to the
    :class:`Conversation` constructor *before* binding so that the
    callback is part of the composed callback chain from the very first
    event.
    """

    def __init__(self, max_cost_per_instance: float, instance_id: str) -> None:
        """
        Args:
            max_cost_per_instance: Maximum allowed accumulated USD cost for
                this instance. Must be strictly positive.
            instance_id: Identifier used only for log messages.

        Raises:
            ValueError: If ``max_cost_per_instance`` is not strictly positive.
        """
        if max_cost_per_instance <= 0:
            raise ValueError(
                f"max_cost_per_instance must be > 0, got {max_cost_per_instance}"
            )
        self.max_cost_per_instance = max_cost_per_instance
        self.instance_id = instance_id
        self._conversation = None  # type: ignore[assignment]
        self._triggered = False

    def bind(self, conversation: object) -> None:
        """Attach the conversation whose cost will be monitored."""
        self._conversation = conversation

    def __call__(self, event: Event) -> None:  # noqa: ARG002 - event is unused
        if self._triggered or self._conversation is None:
            return
        try:
            cost = self._conversation.conversation_stats.get_combined_metrics().accumulated_cost
        except Exception as exc:
            # Metrics access should never block the run.
            logger.debug(
                "cost_cap: failed to read accumulated_cost for %s: %s",
                self.instance_id,
                exc,
            )
            return

        if cost >= self.max_cost_per_instance:
            self._triggered = True
            logger.warning(
                "cost_cap: instance %s exceeded per-instance budget "
                "(accumulated_cost=$%.4f >= cap=$%.4f); pausing conversation.",
                self.instance_id,
                cost,
                self.max_cost_per_instance,
            )
            try:
                self._conversation.pause()
            except Exception as exc:
                # Defensive: if pause itself fails (e.g. remote conversation
                # in an odd state), don't take the whole instance down.
                logger.warning(
                    "cost_cap: pause() raised for %s: %s",
                    self.instance_id,
                    exc,
                )


def build_cost_cap_callback(
    max_cost_per_instance: float, instance_id: str
) -> CostCapCallback:
    """Convenience wrapper. See :class:`CostCapCallback`.

    Returns:
        An unbound :class:`CostCapCallback`. The caller must call
        ``bind(conversation)`` once the conversation has been created so
        the callback knows which conversation to read cost from and pause.
    """
    return CostCapCallback(
        max_cost_per_instance=max_cost_per_instance,
        instance_id=instance_id,
    )
