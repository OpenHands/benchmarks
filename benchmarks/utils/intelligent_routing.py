"""Intelligent per-instance model routing for benchmarks.

A *router config* is a JSON file shaped like::

    {
      "kind": "intelligent-router-v0",
      "classifier_model_id": "minimax-m2.7",
      "fallback_model_id": "gpt-5.5",
      "tiers": {
        "kimi-k2.6":    { ...standard LLM config... },
        "minimax-m2.7": { ...standard LLM config... },
        "gpt-5.5":      { ...standard LLM config... }
      },
      "routing": {
        "Frontend":                 "kimi-k2.6",
        "Issue Resolution (other)": "minimax-m2.7",
        "Greenfield":               "gpt-5.5",
        "Testing":                  "gpt-5.5",
        "Information Gathering":    "gpt-5.5"
      }
    }

When a benchmark sees this shape, it calls the classifier LLM once per
instance against ``CLASSIFIER_PROMPT`` to pick one of the categories above,
then runs the conversation on the matching tier LLM. The decision is logged
per instance for offline analysis.

This module does **not** depend on any benchmark-specific code; benchmarks
only call ``classify_and_route`` with their own task-text extractor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from openhands.sdk import LLM, get_logger
from openhands.sdk.llm import Message, TextContent


logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Classifier prompt (iter5, frozen verbatim from research repo)                #
# --------------------------------------------------------------------------- #

# Source: OpenHands/research/juan/intelligent-model-selection/
#         optimization_200_4cat/classifier_prompt_iter5.txt
CLASSIFIER_PROMPT = """\
You are a task classification expert. Your job is to classify software development task instructions into exactly one of the following categories:

**Categories:**

1. **Greenfield** - Tasks that involve creating new projects, repositories, or applications from scratch. These tasks typically start with nothing and build something new.

2. **Frontend** - Tasks focused on user interface, visual rendering, styling, or frontend component behavior issues. Key indicators: UI components, forms, visual display, CSS, rendering problems, dynamic UI behavior, frontend frameworks, visual elements, or browser-specific issues. Must explicitly mention UI/visual/display/rendering/styling concerns.

3. **Testing** - Tasks involving problems with test logic, test computations, test utilities, or testing infrastructure. Key indicators: test functions computing wrong values, test methods producing incorrect results, test utilities with bugs, testing framework malfunctions, or phrases describing how tests themselves are broken or compute incorrectly.

4. **Information Gathering** - Tasks that involve research, questions, or gathering information WITHOUT any implementation, fixing, or changes. Pure information requests only. Key phrases: "Should we...", "Consider whether...", "What if...", "Is it better to..." when asking for opinions rather than reporting bugs.

5. **Issue Resolution (other)** - All other bug fixes, issue resolution, and debugging tasks including: backend logic bugs, algorithms, data processing, APIs, core functionality, model computations, library functions, data structures, configuration issues, development tool bugs, computational problems, incorrect calculations, wrong outputs, data handling errors, parser issues, linter bugs, and any functional problems.

**Classification Rules:**

1. **Bug reports are Issue Resolution by default** - If a task describes something not working correctly, producing wrong results, or behaving incorrectly, classify as Issue Resolution (other) UNLESS it explicitly describes UI/visual problems (Frontend) or test infrastructure problems (Testing).

2. **Testing**: Only for broken test infrastructure itself. If underlying code produces wrong results and tests detect it, that's Issue Resolution.

3. **Frontend**: Requires explicit mention of UI, visual, display, rendering, styling, or frontend framework components. Generic "bug" or "error" without UI context is Issue Resolution.

4. **Tool/library bugs are Issue Resolution** - ESLint issues, parser problems, linter bugs, configuration errors, and development tool malfunctions are Issue Resolution (other), not Frontend.

5. **Questions about design decisions** - Phrases like "Should we", "Consider whether", "What if" asking for opinions are Information Gathering only if no bug is being reported.

Respond with ONLY the category name, exactly as written above.
"""

# Canonical category strings the classifier may emit. Order matters for the
# substring matcher below: longer / more specific keys appear first.
CATEGORIES: list[str] = [
    "Issue Resolution (other)",
    "Information Gathering",
    "Greenfield",
    "Frontend",
    "Testing",
]


# --------------------------------------------------------------------------- #
# Default routing table (the user-specified 3-tier mapping)                    #
# --------------------------------------------------------------------------- #

DEFAULT_ROUTING: dict[str, str] = {
    "Frontend": "kimi-k2.6",
    "Issue Resolution (other)": "minimax-m2.7",
    "Greenfield": "gpt-5.5",
    "Testing": "gpt-5.5",
    "Information Gathering": "gpt-5.5",
}

# Model IDs known to accept image inputs. Used by ``classify_and_route`` to
# fall back to a vision-capable tier when an instance has images but the
# classified tier is text-only (e.g. swebenchmultimodal frontend instance
# routed to a text-only tier). Extend as new tiers are added.
DEFAULT_VISION_CAPABLE: frozenset[str] = frozenset({"kimi-k2.6", "gpt-5.5"})


# --------------------------------------------------------------------------- #
# Per-benchmark task-text extractors                                           #
# --------------------------------------------------------------------------- #

BenchmarkName = Literal[
    "swebench",
    "swebenchmultimodal",
    "swtbench",
    "gaia",
    "commit0",
]

TaskTextExtractor = Callable[[dict], str]


def _swebench_task_text(data: dict) -> str:
    return str(data.get("problem_statement", "") or "")


def _gaia_task_text(data: dict) -> str:
    return str(data.get("Question", "") or "")


def _commit0_task_text(data: dict) -> str:
    # commit0 instances render the spec into the agent prompt; the spec text
    # itself isn't always denormalized onto ``data``. Fall back to
    # ``problem_statement`` and then to the repo name so the classifier has
    # *something* to work with rather than emitting EMPTY_TASK across the
    # whole benchmark.
    for key in ("spec", "problem_statement", "instruction", "issue"):
        value = data.get(key)
        if value:
            return str(value)
    repo = data.get("repo", "")
    return f"Implement the project from its spec: {repo}" if repo else ""


BENCHMARK_TASK_EXTRACTORS: dict[BenchmarkName, TaskTextExtractor] = {
    "swebench": _swebench_task_text,
    "swebenchmultimodal": _swebench_task_text,
    "swtbench": _swebench_task_text,
    "gaia": _gaia_task_text,
    "commit0": _commit0_task_text,
}


def _instance_has_images(data: dict) -> bool:
    """Best-effort detection of image-bearing instances (swebenchmultimodal)."""
    assets = data.get("image_assets")
    if not assets:
        return False
    if isinstance(assets, str):
        try:
            assets = json.loads(assets)
        except (TypeError, ValueError):
            return False
    if not isinstance(assets, dict):
        return False
    return bool(assets.get("problem_statement"))


# --------------------------------------------------------------------------- #
# Router spec + decision types                                                 #
# --------------------------------------------------------------------------- #


class RouterSpec(BaseModel):
    """Parsed intelligent-router-v0 configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    classifier_llm: LLM = Field(
        description=(
            "LLM used to classify each instance into a category. "
            "Typically one of the tier LLMs (e.g. minimax-m2.7)."
        ),
    )
    tiers: dict[str, LLM] = Field(
        description="Map of model_id -> LLM instance for each routing tier.",
    )
    routing: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_ROUTING),
        description="Map of classifier category -> tier model_id.",
    )
    fallback_model_id: str = Field(
        description=(
            "Model ID used when classification is empty, unparseable, or the "
            "chosen tier is text-only but the instance has images."
        ),
    )
    vision_capable_model_ids: set[str] = Field(
        default_factory=lambda: set(DEFAULT_VISION_CAPABLE),
        description="Model IDs known to accept image inputs.",
    )

    def tier_or_fallback(self, model_id: str) -> tuple[str, LLM]:
        """Return ``(model_id, llm)``, falling back if the ID is unknown."""
        if model_id in self.tiers:
            return model_id, self.tiers[model_id]
        return self.fallback_model_id, self.tiers[self.fallback_model_id]


@dataclass(frozen=True)
class RoutingDecision:
    """Result of classifying and routing a single instance."""

    chosen_llm: LLM
    chosen_model_id: str
    category: str
    raw_classifier_output: str
    forced_vision_fallback: bool


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


_ROUTER_KIND = "intelligent-router-v0"


def is_router_config_payload(payload: dict) -> bool:
    """Return True if the given parsed JSON object is a router config."""
    return isinstance(payload, dict) and payload.get("kind") == _ROUTER_KIND


def maybe_load_router_spec(config_path: str | Path) -> RouterSpec | None:
    """Load a router config from ``config_path`` if it matches the v0 shape.

    Returns ``None`` for plain LLM configs (so the caller can fall back to
    ``load_llm_config``). Raises ``ValueError`` if the payload claims to be a
    router config but is missing required fields.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ValueError(f"LLM config file {path} does not exist")

    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Not JSON or not a router payload; let the plain loader produce its
        # own error.
        return None

    if not is_router_config_payload(payload):
        return None

    return _build_router_spec(payload)


def _build_router_spec(payload: dict) -> RouterSpec:
    tiers_raw = payload.get("tiers")
    if not isinstance(tiers_raw, dict) or not tiers_raw:
        raise ValueError("router config: 'tiers' must be a non-empty object")

    classifier_model_id = payload.get("classifier_model_id")
    if not classifier_model_id or classifier_model_id not in tiers_raw:
        raise ValueError(
            "router config: 'classifier_model_id' must reference a key in 'tiers'"
        )

    fallback_model_id = payload.get("fallback_model_id") or classifier_model_id
    if fallback_model_id not in tiers_raw:
        raise ValueError(
            "router config: 'fallback_model_id' must reference a key in 'tiers'"
        )

    tiers: dict[str, LLM] = {}
    for model_id, llm_cfg in tiers_raw.items():
        if not isinstance(llm_cfg, dict):
            raise ValueError(f"router config: tier '{model_id}' must be an object")
        # Attach a stable usage_id so per-tier metrics are distinguishable in
        # traces and per-instance cost reports.
        cfg = {**llm_cfg, "usage_id": llm_cfg.get("usage_id") or f"agent:{model_id}"}
        tiers[model_id] = LLM.model_validate(cfg)

    classifier_cfg = {
        **tiers_raw[classifier_model_id],
        "usage_id": "router:classifier",
    }
    classifier_llm = LLM.model_validate(classifier_cfg)

    routing = payload.get("routing") or DEFAULT_ROUTING
    if not isinstance(routing, dict) or not routing:
        raise ValueError("router config: 'routing' must be a non-empty object")
    unknown_targets = {v for v in routing.values() if v not in tiers}
    if unknown_targets:
        raise ValueError(
            f"router config: routing targets not present in 'tiers': {unknown_targets}"
        )

    vision = payload.get("vision_capable_model_ids")
    vision_set: set[str] = (
        set(vision) if isinstance(vision, list) else set(DEFAULT_VISION_CAPABLE)
    )

    return RouterSpec(
        classifier_llm=classifier_llm,
        tiers=tiers,
        routing=routing,
        fallback_model_id=fallback_model_id,
        vision_capable_model_ids=vision_set,
    )


def parse_classifier_output(raw: str, routing: dict[str, str]) -> str | None:
    """Map a raw classifier response to a routing category, or ``None``.

    Substring-matches against the canonical category strings in priority order
    (most specific first). Case-insensitive. Tolerates models that wrap their
    answer in markdown or commentary.
    """
    if not raw:
        return None
    lowered = raw.lower()
    for category in CATEGORIES:
        if category.lower() in lowered:
            return category if category in routing else None
    # Loose fallback: bare keywords without the "(other)" suffix.
    for keyword, canonical in (
        ("issue resolution", "Issue Resolution (other)"),
        ("information gathering", "Information Gathering"),
        ("greenfield", "Greenfield"),
        ("frontend", "Frontend"),
        ("testing", "Testing"),
    ):
        if keyword in lowered and canonical in routing:
            return canonical
    return None


def classify_and_route(
    benchmark: BenchmarkName | str,
    instance_data: dict,
    router: RouterSpec,
) -> RoutingDecision:
    """Classify a single instance and pick the matching tier LLM.

    Falls back to ``router.fallback_model_id`` when:

    * the task text is empty,
    * the classifier output cannot be parsed into a known category,
    * the instance carries images but the chosen tier isn't vision-capable.

    All cost-relevant LLM construction happens here (via the LLM objects
    stored on the router); per-instance virtual-key injection should still be
    applied by the caller via :func:`benchmarks.utils.litellm_proxy.build_eval_llm`.
    """
    extractor = BENCHMARK_TASK_EXTRACTORS.get(benchmark)  # type: ignore[arg-type]
    if extractor is None:
        chosen_id, chosen_llm = router.tier_or_fallback(router.fallback_model_id)
        return RoutingDecision(
            chosen_llm=chosen_llm,
            chosen_model_id=chosen_id,
            category="NO_EXTRACTOR",
            raw_classifier_output="",
            forced_vision_fallback=False,
        )

    task_text = extractor(instance_data).strip()
    if not task_text:
        chosen_id, chosen_llm = router.tier_or_fallback(router.fallback_model_id)
        return RoutingDecision(
            chosen_llm=chosen_llm,
            chosen_model_id=chosen_id,
            category="EMPTY_TASK",
            raw_classifier_output="",
            forced_vision_fallback=False,
        )

    raw = _run_classifier(router.classifier_llm, task_text)
    category = parse_classifier_output(raw, router.routing)

    if category is None:
        chosen_id, chosen_llm = router.tier_or_fallback(router.fallback_model_id)
        return RoutingDecision(
            chosen_llm=chosen_llm,
            chosen_model_id=chosen_id,
            category="UNPARSED",
            raw_classifier_output=raw,
            forced_vision_fallback=False,
        )

    target_id = router.routing[category]
    chosen_id, chosen_llm = router.tier_or_fallback(target_id)

    forced = False
    if (
        _instance_has_images(instance_data)
        and chosen_id not in router.vision_capable_model_ids
    ):
        forced = True
        chosen_id, chosen_llm = router.tier_or_fallback(router.fallback_model_id)
        if chosen_id not in router.vision_capable_model_ids:
            # Fallback isn't vision-capable either: try the first declared one.
            for candidate in router.vision_capable_model_ids:
                if candidate in router.tiers:
                    chosen_id = candidate
                    chosen_llm = router.tiers[candidate]
                    break

    return RoutingDecision(
        chosen_llm=chosen_llm,
        chosen_model_id=chosen_id,
        category=category,
        raw_classifier_output=raw,
        forced_vision_fallback=forced,
    )


def _run_classifier(classifier_llm: LLM, task_text: str) -> str:
    """Run the classifier prompt against a single task; return raw text."""
    try:
        response = classifier_llm.completion(
            messages=[
                Message(role="system", content=[TextContent(text=CLASSIFIER_PROMPT)]),
                Message(role="user", content=[TextContent(text=task_text)]),
            ]
        )
    except Exception as exc:  # noqa: BLE001 — best-effort classification
        logger.warning("Classifier call failed: %s", exc, exc_info=True)
        return ""

    message = getattr(response, "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text")
            if text:
                parts.append(str(text))
        return "".join(parts).strip()
    return ""
