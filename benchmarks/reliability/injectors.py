"""Concrete dispatch from declarative faults to explicit runtime capabilities."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from benchmarks.reliability.models import (
    EventRecord,
    FaultKind,
    FaultReceipt,
    FaultSpec,
    JsonObject,
    ReceiptStatus,
    RunIdentity,
)
from benchmarks.reliability.schedule import FaultSchedule


class FaultInjectionError(RuntimeError):
    """Raised after a failed injection has emitted an inspectable receipt."""


class FaultContext(Protocol):
    """Infrastructure capabilities supplied by one benchmark adapter."""

    @property
    def run(self) -> RunIdentity:
        """Return the active run identity."""
        ...

    def monotonic(self) -> float:
        """Return the adapter's monotonic clock."""
        ...

    def restart_sandbox(self, fault: FaultSpec) -> JsonObject:
        """Restart and reattach the scenario's real sandbox/runtime."""
        ...

    def drop_dispatch_response(self, fault: FaultSpec) -> JsonObject:
        """Drop a response at the adapter's verified dispatch boundary."""
        ...

    def sigkill_mid_tool_call(self, fault: FaultSpec) -> JsonObject:
        """SIGKILL the adapter's tool process/container while it is active."""
        ...

    def partition_network(self, fault: FaultSpec) -> JsonObject:
        """Apply the declared network partition at a named endpoint boundary."""
        ...

    def heal_network(self, fault: FaultSpec) -> JsonObject:
        """Release a previously applied network partition."""
        ...


@dataclass(frozen=True, slots=True)
class FaultHandle:
    """Armed fault state retained until optional release."""

    fault: FaultSpec
    trigger_event: EventRecord


class FaultInjector(Protocol):
    """Apply and optionally release one fault category."""

    def inject(self, context: FaultContext, handle: FaultHandle) -> FaultReceipt:
        """Apply the armed fault and return inspectable evidence."""
        ...

    def release(
        self,
        context: FaultContext,
        handle: FaultHandle,
    ) -> FaultReceipt | None:
        """Release a bounded fault, if required."""
        ...


class _OneShotInjector:
    def __init__(
        self,
        operation: Callable[[FaultContext, FaultSpec], JsonObject],
    ) -> None:
        self._operation = operation

    def inject(self, context: FaultContext, handle: FaultHandle) -> FaultReceipt:
        details = self._operation(context, handle.fault)
        return _receipt(
            context,
            handle,
            status=ReceiptStatus.APPLIED,
            details=details,
        )

    def release(
        self,
        context: FaultContext,
        handle: FaultHandle,
    ) -> FaultReceipt | None:
        return None


class _NetworkPartitionInjector:
    def inject(self, context: FaultContext, handle: FaultHandle) -> FaultReceipt:
        details = context.partition_network(handle.fault)
        return _receipt(
            context,
            handle,
            status=ReceiptStatus.APPLIED,
            details=details,
        )

    def release(
        self,
        context: FaultContext,
        handle: FaultHandle,
    ) -> FaultReceipt:
        details = context.heal_network(handle.fault)
        return _receipt(
            context,
            handle,
            status=ReceiptStatus.RELEASED,
            details=details,
        )


class FaultController:
    """Match persisted events, inject faults, and preserve every receipt."""

    def __init__(
        self,
        *,
        schedule: FaultSchedule,
        context: FaultContext,
        on_receipt: Callable[[FaultReceipt], None],
    ) -> None:
        self._schedule = schedule
        self._context = context
        self._on_receipt = on_receipt
        self._active: list[tuple[FaultInjector, FaultHandle]] = []

    def observe(self, event: EventRecord) -> tuple[FaultReceipt, ...]:
        """Inject every newly matched fault after its trigger event is persisted."""
        receipts: list[FaultReceipt] = []
        for fault in self._schedule.observe((event,)):
            handle = FaultHandle(fault=fault, trigger_event=event)
            injector = injector_for(fault)
            try:
                receipt = injector.inject(self._context, handle)
            except Exception as exc:
                receipt = _receipt(
                    self._context,
                    handle,
                    status=ReceiptStatus.FAILED,
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
                self._on_receipt(receipt)
                raise FaultInjectionError(f"failed to inject {fault.fault_id}") from exc
            self._on_receipt(receipt)
            receipts.append(receipt)
            self._active.append((injector, handle))
        return tuple(receipts)

    def release_all(self) -> tuple[FaultReceipt, ...]:
        """Release bounded faults in reverse application order."""
        receipts: list[FaultReceipt] = []
        while self._active:
            injector, handle = self._active.pop()
            try:
                receipt = injector.release(self._context, handle)
            except Exception as exc:
                receipt = _receipt(
                    self._context,
                    handle,
                    status=ReceiptStatus.FAILED,
                    details={
                        "operation": "release",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                self._on_receipt(receipt)
                raise FaultInjectionError(
                    f"failed to release {handle.fault.fault_id}"
                ) from exc
            if receipt is not None:
                self._on_receipt(receipt)
                receipts.append(receipt)
        return tuple(receipts)


def injector_for(fault: FaultSpec) -> FaultInjector:
    """Resolve a concrete injector for a supported fault specification."""
    if fault.kind == FaultKind.SANDBOX_RESTART:
        return _OneShotInjector(lambda context, spec: context.restart_sandbox(spec))
    if fault.kind == FaultKind.LOST_DISPATCH_RESPONSE:
        return _OneShotInjector(
            lambda context, spec: context.drop_dispatch_response(spec)
        )
    if fault.kind == FaultKind.SIGKILL_MID_TOOL_CALL:
        return _OneShotInjector(
            lambda context, spec: context.sigkill_mid_tool_call(spec)
        )
    if fault.kind == FaultKind.NETWORK_PARTITION:
        return _NetworkPartitionInjector()
    raise AssertionError(f"unhandled fault kind: {fault.kind}")


def _receipt(
    context: FaultContext,
    handle: FaultHandle,
    *,
    status: ReceiptStatus,
    details: JsonObject,
) -> FaultReceipt:
    return FaultReceipt(
        run_id=context.run.run_id,
        fault_id=handle.fault.fault_id,
        kind=handle.fault.kind,
        status=status,
        monotonic_seconds=context.monotonic(),
        event_id=handle.trigger_event.event_id,
        tool_call_id=handle.trigger_event.tool_call_id,
        details=details,
    )
