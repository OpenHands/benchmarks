"""Tests for the per-instance cost cap callback."""

from __future__ import annotations

import pytest

from benchmarks.utils.cost_cap import CostCapCallback, build_cost_cap_callback


class _FakeMetrics:
    def __init__(self, cost: float) -> None:
        self.accumulated_cost = cost


class _FakeStats:
    def __init__(self, cost: float) -> None:
        self._cost = cost

    def get_combined_metrics(self) -> _FakeMetrics:
        return _FakeMetrics(self._cost)


class _FakeConversation:
    """Minimal stand-in for ``BaseConversation`` for unit testing.

    Exposes only the surface the cost-cap callback touches:
    ``conversation_stats`` and ``pause()``.
    """

    def __init__(self, cost: float) -> None:
        self._cost = cost
        self.paused = False
        self.pause_call_count = 0

    @property
    def conversation_stats(self) -> _FakeStats:
        return _FakeStats(self._cost)

    def set_cost(self, cost: float) -> None:
        self._cost = cost

    def pause(self) -> None:
        self.paused = True
        self.pause_call_count += 1


def test_rejects_non_positive_cap():
    with pytest.raises(ValueError):
        build_cost_cap_callback(max_cost_per_instance=0.0, instance_id="x")
    with pytest.raises(ValueError):
        build_cost_cap_callback(max_cost_per_instance=-1.0, instance_id="x")


def test_no_pause_below_cap():
    convo = _FakeConversation(cost=2.5)
    cb = build_cost_cap_callback(max_cost_per_instance=10.0, instance_id="inst")
    cb.bind(convo)
    cb(event=object())  # event is unused
    assert convo.paused is False


def test_pauses_when_cap_reached():
    convo = _FakeConversation(cost=10.0)
    cb = build_cost_cap_callback(max_cost_per_instance=10.0, instance_id="inst")
    cb.bind(convo)
    cb(event=object())
    assert convo.paused is True
    assert convo.pause_call_count == 1


def test_pauses_when_cap_exceeded():
    convo = _FakeConversation(cost=0.0)
    cb = build_cost_cap_callback(max_cost_per_instance=5.0, instance_id="inst")
    cb.bind(convo)
    cb(event=object())
    assert convo.paused is False

    convo.set_cost(7.5)
    cb(event=object())
    assert convo.paused is True


def test_idempotent_once_triggered():
    """Once paused, the callback must not call pause() again on subsequent events."""
    convo = _FakeConversation(cost=20.0)
    cb = build_cost_cap_callback(max_cost_per_instance=5.0, instance_id="inst")
    cb.bind(convo)
    for _ in range(5):
        cb(event=object())
    assert convo.pause_call_count == 1


def test_no_op_when_not_bound():
    """Calling the callback before bind() should be a safe no-op."""
    cb = build_cost_cap_callback(max_cost_per_instance=1.0, instance_id="inst")
    # Should not raise.
    cb(event=object())
    # Still works after binding.
    convo = _FakeConversation(cost=2.0)
    cb.bind(convo)
    cb(event=object())
    assert convo.paused is True


def test_metrics_failure_does_not_crash():
    """If reading metrics raises, the callback must swallow the error."""

    class _BrokenConversation(_FakeConversation):
        @property
        def conversation_stats(self):
            raise RuntimeError("metrics unavailable")

    convo = _BrokenConversation(cost=0.0)
    cb = build_cost_cap_callback(max_cost_per_instance=1.0, instance_id="inst")
    cb.bind(convo)
    cb(event=object())  # must not raise
    assert convo.paused is False


def test_pause_failure_does_not_crash():
    """If pause() raises, the callback must swallow the error and stay triggered."""

    class _BrokenPauseConversation(_FakeConversation):
        def pause(self) -> None:
            self.pause_call_count += 1
            raise RuntimeError("cannot pause")

    convo = _BrokenPauseConversation(cost=10.0)
    cb = build_cost_cap_callback(max_cost_per_instance=1.0, instance_id="inst")
    cb.bind(convo)
    cb(event=object())  # must not raise
    # Once it tried to pause once, it stays triggered and won't try again.
    cb(event=object())
    assert convo.pause_call_count == 1


def test_callback_class_directly():
    """CostCapCallback can be constructed and used directly."""
    convo = _FakeConversation(cost=15.0)
    cb = CostCapCallback(max_cost_per_instance=10.0, instance_id="direct")
    cb.bind(convo)
    cb(event=object())
    assert convo.paused is True
