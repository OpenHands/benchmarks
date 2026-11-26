"""Scenario document loading with a strict, versioned schema."""

import json
from pathlib import Path
from typing import cast

from benchmarks.reliability.models import (
    EventTrigger,
    FaultKind,
    FaultSpec,
    JsonObject,
    JsonValue,
    Scenario,
)


def load_scenario(path: Path) -> Scenario:
    """Load a JSON scenario and reject unknown or malformed fields."""
    if path.suffix.lower() != ".json":
        raise ValueError("scenario path must end in .json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("scenario document must contain a mapping")
    return scenario_from_mapping(cast(JsonObject, raw))


def scenario_from_mapping(raw: JsonObject) -> Scenario:
    """Build a scenario while rejecting unknown fields at every level."""
    _require_exact_keys(
        raw,
        required={
            "schema_version",
            "scenario_id",
            "task",
            "agent",
            "seed",
            "faults",
        },
        location="scenario",
    )
    task = _mapping(raw["task"], "task")
    _require_exact_keys(task, required={"benchmark", "instance_id"}, location="task")
    agent = _mapping(raw["agent"], "agent")
    _require_exact_keys(agent, required={"config"}, location="agent")
    faults_raw = raw["faults"]
    if not isinstance(faults_raw, list):
        raise ValueError("faults must be a list")

    return Scenario(
        schema_version=_integer(raw["schema_version"], "schema_version"),
        scenario_id=_string(raw["scenario_id"], "scenario_id"),
        benchmark=_string(task["benchmark"], "task.benchmark"),
        instance_id=_string(task["instance_id"], "task.instance_id"),
        agent_config=_string(agent["config"], "agent.config"),
        seed=_integer(raw["seed"], "seed"),
        faults=tuple(
            _fault_from_mapping(_mapping(item, f"faults[{index}]"), index)
            for index, item in enumerate(faults_raw)
        ),
    )


def _fault_from_mapping(raw: JsonObject, index: int) -> FaultSpec:
    location = f"faults[{index}]"
    _require_exact_keys(
        raw,
        required={"fault_id", "kind", "trigger"},
        optional={"parameters"},
        location=location,
    )
    trigger_wrapper = _mapping(raw["trigger"], f"{location}.trigger")
    _require_exact_keys(
        trigger_wrapper,
        required={"event"},
        location=f"{location}.trigger",
    )
    event = _mapping(trigger_wrapper["event"], f"{location}.trigger.event")
    _require_exact_keys(
        event,
        required={"event_type", "ordinal"},
        optional={"tool_call_id"},
        location=f"{location}.trigger.event",
    )
    parameters_raw = raw.get("parameters", {})
    parameters = _mapping(parameters_raw, f"{location}.parameters")
    tool_call_id_raw = event.get("tool_call_id")
    tool_call_id = (
        None
        if tool_call_id_raw is None
        else _string(tool_call_id_raw, f"{location}.trigger.event.tool_call_id")
    )
    try:
        kind = FaultKind(_string(raw["kind"], f"{location}.kind"))
    except ValueError as exc:
        supported = ", ".join(item.value for item in FaultKind)
        raise ValueError(f"{location}.kind must be one of: {supported}") from exc

    return FaultSpec(
        fault_id=_string(raw["fault_id"], f"{location}.fault_id"),
        kind=kind,
        trigger=EventTrigger(
            event_type=_string(
                event["event_type"],
                f"{location}.trigger.event.event_type",
            ),
            ordinal=_integer(
                event["ordinal"],
                f"{location}.trigger.event.ordinal",
            ),
            tool_call_id=tool_call_id,
        ),
        parameters=parameters,
    )


def _mapping(value: JsonValue | object, location: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    return cast(JsonObject, value)


def _string(value: JsonValue, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _integer(value: JsonValue, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    return value


def _require_exact_keys(
    value: JsonObject,
    *,
    required: set[str],
    optional: set[str] | None = None,
    location: str,
) -> None:
    optional = optional or set()
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ValueError(f"{location} missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{location} unknown fields: {', '.join(sorted(unknown))}")
