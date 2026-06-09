"""Tests for benchmarks.utils.intelligent_routing.

These tests cover the routing helpers directly without making real LLM
calls. The classifier is exercised end-to-end with a tiny in-memory LLM
stub so that ``classify_and_route`` exercises real parsing/dispatch code
paths rather than a mock of itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.utils.intelligent_routing import (
    BENCHMARK_TASK_EXTRACTORS,
    DEFAULT_ROUTING,
    RouterSpec,
    RoutingDecision,
    classify_and_route,
    is_router_config_payload,
    maybe_load_router_spec,
    parse_classifier_output,
)
from openhands.sdk import LLM


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _llm(model: str, **extra: Any) -> LLM:
    return LLM.model_validate({"model": model, **extra})


def _router_spec(**overrides: Any) -> RouterSpec:
    tiers = {
        "kimi-k2.6": _llm("litellm_proxy/moonshot/kimi-k2.6"),
        "minimax-m2.7": _llm("litellm_proxy/minimax/MiniMax-M2.7"),
        "gpt-5.5": _llm("litellm_proxy/openai/gpt-5.5"),
    }
    spec = RouterSpec(
        classifier_llm=tiers["minimax-m2.7"],
        tiers=tiers,
        routing=dict(DEFAULT_ROUTING),
        fallback_model_id="gpt-5.5",
        vision_capable_model_ids={"kimi-k2.6", "gpt-5.5"},
    )
    if overrides:
        spec = spec.model_copy(update=overrides)
    return spec


class _StubClassifier:
    """Captures classifier calls and returns a predetermined response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[Any]] = []

    def completion(self, messages: list[Any], **_: Any) -> Any:  # noqa: ANN401
        self.calls.append(messages)

        class _Part:
            def __init__(self, text: str) -> None:
                self.text = text

        class _Msg:
            def __init__(self, text: str) -> None:
                self.content = [_Part(text)]

        class _Resp:
            def __init__(self, text: str) -> None:
                self.message = _Msg(text)

        return _Resp(self.response)


def _attach_stub(spec: RouterSpec, response: str) -> _StubClassifier:
    stub = _StubClassifier(response)
    # Pydantic models are frozen-by-validation, but ``classifier_llm`` is a
    # plain attribute; mutate it directly for the test stub.
    object.__setattr__(spec, "classifier_llm", stub)
    return stub


# --------------------------------------------------------------------------- #
# Output parsing                                                               #
# --------------------------------------------------------------------------- #


class TestParseClassifierOutput:
    def test_exact_match(self) -> None:
        assert parse_classifier_output("Frontend", DEFAULT_ROUTING) == "Frontend"

    def test_exact_match_issue_resolution_other(self) -> None:
        assert (
            parse_classifier_output("Issue Resolution (other)", DEFAULT_ROUTING)
            == "Issue Resolution (other)"
        )

    def test_case_insensitive(self) -> None:
        assert parse_classifier_output("FRONTEND", DEFAULT_ROUTING) == "Frontend"

    def test_chatty_response_returns_category(self) -> None:
        raw = "**Frontend** — this clearly involves CSS rendering."
        assert parse_classifier_output(raw, DEFAULT_ROUTING) == "Frontend"

    def test_bare_keyword_resolves_to_canonical_with_suffix(self) -> None:
        # Model emitted the bare keyword without the "(other)" suffix.
        assert (
            parse_classifier_output("Issue Resolution", DEFAULT_ROUTING)
            == "Issue Resolution (other)"
        )

    def test_unknown_category_returns_none(self) -> None:
        assert parse_classifier_output("Backend Refactor", DEFAULT_ROUTING) is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_classifier_output("", DEFAULT_ROUTING) is None

    def test_most_specific_match_wins(self) -> None:
        # "Information Gathering" must beat "Greenfield" if both appear,
        # because they are checked in priority order and "Information
        # Gathering" appears earlier in CATEGORIES.
        raw = "Information Gathering. (Could be Greenfield but no.)"
        assert parse_classifier_output(raw, DEFAULT_ROUTING) == "Information Gathering"


# --------------------------------------------------------------------------- #
# Router config loading                                                        #
# --------------------------------------------------------------------------- #


def _write_router_config(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "router.json"
    path.write_text(json.dumps(payload))
    return path


def _minimal_router_payload() -> dict[str, Any]:
    return {
        "kind": "intelligent-router-v0",
        "classifier_model_id": "minimax-m2.7",
        "fallback_model_id": "gpt-5.5",
        "tiers": {
            "kimi-k2.6": {"model": "litellm_proxy/moonshot/kimi-k2.6"},
            "minimax-m2.7": {"model": "litellm_proxy/minimax/MiniMax-M2.7"},
            "gpt-5.5": {"model": "litellm_proxy/openai/gpt-5.5"},
        },
        "routing": dict(DEFAULT_ROUTING),
    }


class TestRouterConfigLoading:
    def test_is_router_config_payload_positive(self) -> None:
        assert is_router_config_payload({"kind": "intelligent-router-v0"})

    def test_is_router_config_payload_negative(self) -> None:
        assert not is_router_config_payload({"model": "gpt-4o"})
        assert not is_router_config_payload({})
        assert not is_router_config_payload({"kind": "something-else"})

    def test_load_minimal_config(self, tmp_path: Path) -> None:
        path = _write_router_config(tmp_path, _minimal_router_payload())

        spec = maybe_load_router_spec(path)

        assert spec is not None
        assert set(spec.tiers) == {"kimi-k2.6", "minimax-m2.7", "gpt-5.5"}
        assert spec.classifier_llm.model == "litellm_proxy/minimax/MiniMax-M2.7"
        assert spec.fallback_model_id == "gpt-5.5"
        assert spec.routing == DEFAULT_ROUTING

    def test_load_assigns_per_tier_usage_id(self, tmp_path: Path) -> None:
        path = _write_router_config(tmp_path, _minimal_router_payload())

        spec = maybe_load_router_spec(path)

        assert spec is not None
        assert spec.tiers["kimi-k2.6"].usage_id == "agent:kimi-k2.6"
        assert spec.tiers["gpt-5.5"].usage_id == "agent:gpt-5.5"
        assert spec.classifier_llm.usage_id == "router:classifier"

    def test_plain_llm_config_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "plain.json"
        path.write_text(json.dumps({"model": "gpt-4o"}))

        assert maybe_load_router_spec(path) is None

    def test_malformed_json_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json")

        # Delegated to the plain loader, which raises its own ValidationError.
        assert maybe_load_router_spec(path) is None

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            maybe_load_router_spec(tmp_path / "nope.json")

    def test_missing_classifier_model_id_raises(self, tmp_path: Path) -> None:
        payload = _minimal_router_payload()
        payload["classifier_model_id"] = "not-in-tiers"
        path = _write_router_config(tmp_path, payload)

        with pytest.raises(ValueError, match="classifier_model_id"):
            maybe_load_router_spec(path)

    def test_routing_target_not_in_tiers_raises(self, tmp_path: Path) -> None:
        payload = _minimal_router_payload()
        payload["routing"]["Frontend"] = "phantom-model"
        path = _write_router_config(tmp_path, payload)

        with pytest.raises(ValueError, match="routing targets"):
            maybe_load_router_spec(path)

    def test_empty_tiers_raises(self, tmp_path: Path) -> None:
        payload = _minimal_router_payload()
        payload["tiers"] = {}
        path = _write_router_config(tmp_path, payload)

        with pytest.raises(ValueError, match="tiers"):
            maybe_load_router_spec(path)


# --------------------------------------------------------------------------- #
# load_llm_config integration                                                  #
# --------------------------------------------------------------------------- #


class TestLoadLLMConfigWithRouter:
    def test_router_config_returns_classifier_llm(self, tmp_path: Path) -> None:
        from benchmarks.utils.llm_config import load_llm_config

        path = _write_router_config(tmp_path, _minimal_router_payload())
        llm = load_llm_config(path)
        # Should be the classifier (minimax) LLM since the primary LLM slot
        # is filled with the classifier for ACP/condenser fallback paths.
        assert llm.model == "litellm_proxy/minimax/MiniMax-M2.7"

    def test_plain_config_still_works(self, tmp_path: Path) -> None:
        from benchmarks.utils.llm_config import load_llm_config

        path = tmp_path / "plain.json"
        path.write_text(json.dumps({"model": "gpt-4o"}))
        llm = load_llm_config(path)
        assert llm.model == "gpt-4o"


# --------------------------------------------------------------------------- #
# Per-benchmark task-text extraction                                           #
# --------------------------------------------------------------------------- #


class TestTaskTextExtractors:
    def test_swebench_extractor(self) -> None:
        text = BENCHMARK_TASK_EXTRACTORS["swebench"](
            {"problem_statement": "Fix the bug"}
        )
        assert text == "Fix the bug"

    def test_swebench_missing_returns_empty(self) -> None:
        assert BENCHMARK_TASK_EXTRACTORS["swebench"]({}) == ""

    def test_gaia_extractor(self) -> None:
        text = BENCHMARK_TASK_EXTRACTORS["gaia"]({"Question": "What is X?"})
        assert text == "What is X?"

    def test_commit0_prefers_spec(self) -> None:
        text = BENCHMARK_TASK_EXTRACTORS["commit0"](
            {"spec": "Build a calculator", "repo": "example/calc"}
        )
        assert text == "Build a calculator"

    def test_commit0_falls_back_to_repo(self) -> None:
        text = BENCHMARK_TASK_EXTRACTORS["commit0"]({"repo": "foo/bar"})
        assert "foo/bar" in text

    def test_swebenchmultimodal_uses_same_extractor_as_swebench(self) -> None:
        assert (
            BENCHMARK_TASK_EXTRACTORS["swebenchmultimodal"]
            is BENCHMARK_TASK_EXTRACTORS["swebench"]
        )

    def test_swtbench_uses_same_extractor_as_swebench(self) -> None:
        assert (
            BENCHMARK_TASK_EXTRACTORS["swtbench"]
            is BENCHMARK_TASK_EXTRACTORS["swebench"]
        )


# --------------------------------------------------------------------------- #
# classify_and_route end-to-end                                                #
# --------------------------------------------------------------------------- #


class TestClassifyAndRoute:
    def test_frontend_routes_to_kimi(self) -> None:
        spec = _router_spec()
        _attach_stub(spec, "Frontend")

        decision = classify_and_route(
            benchmark="swebench",
            instance_data={
                "problem_statement": "CSS rendering bug on form button hover"
            },
            router=spec,
        )

        assert isinstance(decision, RoutingDecision)
        assert decision.category == "Frontend"
        assert decision.chosen_model_id == "kimi-k2.6"
        assert decision.forced_vision_fallback is False

    def test_issue_resolution_routes_to_minimax(self) -> None:
        spec = _router_spec()
        _attach_stub(spec, "Issue Resolution (other)")

        decision = classify_and_route(
            benchmark="swebench",
            instance_data={"problem_statement": "Pagination off-by-one"},
            router=spec,
        )

        assert decision.category == "Issue Resolution (other)"
        assert decision.chosen_model_id == "minimax-m2.7"

    def test_information_gathering_routes_to_gpt55(self) -> None:
        spec = _router_spec()
        _attach_stub(spec, "Information Gathering")

        decision = classify_and_route(
            benchmark="gaia",
            instance_data={"Question": "How many Olympics has Sweden hosted?"},
            router=spec,
        )

        assert decision.category == "Information Gathering"
        assert decision.chosen_model_id == "gpt-5.5"

    def test_unparseable_response_falls_back(self) -> None:
        spec = _router_spec()
        _attach_stub(spec, "Backend Refactor — not a real category")

        decision = classify_and_route(
            benchmark="swebench",
            instance_data={"problem_statement": "Fix something."},
            router=spec,
        )

        assert decision.category == "UNPARSED"
        assert decision.chosen_model_id == "gpt-5.5"
        assert "Backend Refactor" in decision.raw_classifier_output

    def test_empty_task_text_skips_classifier(self) -> None:
        spec = _router_spec()
        stub = _attach_stub(spec, "Frontend")

        decision = classify_and_route(
            benchmark="swebench",
            instance_data={"problem_statement": ""},
            router=spec,
        )

        assert decision.category == "EMPTY_TASK"
        assert decision.chosen_model_id == "gpt-5.5"
        assert stub.calls == [], "classifier should not be invoked when task is empty"

    def test_unknown_benchmark_uses_fallback(self) -> None:
        spec = _router_spec()

        decision = classify_and_route(
            benchmark="not-a-known-benchmark",  # type: ignore[arg-type]
            instance_data={"problem_statement": "Hello"},
            router=spec,
        )

        assert decision.category == "NO_EXTRACTOR"
        assert decision.chosen_model_id == "gpt-5.5"

    def test_classifier_failure_falls_back_gracefully(self) -> None:
        spec = _router_spec()

        class _ExplodingClassifier:
            def completion(self, **_: Any) -> Any:  # noqa: ANN401
                raise RuntimeError("classifier proxy unavailable")

        object.__setattr__(spec, "classifier_llm", _ExplodingClassifier())

        decision = classify_and_route(
            benchmark="swebench",
            instance_data={"problem_statement": "Something is broken"},
            router=spec,
        )

        # Empty classifier output → UNPARSED → fallback.
        assert decision.category == "UNPARSED"
        assert decision.chosen_model_id == "gpt-5.5"


class TestVisionFallback:
    """When classifier sends an image-bearing instance to a text-only tier,
    routing must redirect to a vision-capable tier (e.g. swebenchmultimodal
    Frontend → kimi is fine; Issue Resolution → minimax is not because
    minimax lacks vision)."""

    def _image_instance(self) -> dict[str, Any]:
        return {
            "problem_statement": "Backend bug; the screenshot shows the error.",
            "image_assets": {"problem_statement": ["https://example.com/a.png"]},
        }

    def test_text_only_tier_for_image_instance_redirects(self) -> None:
        spec = _router_spec()
        _attach_stub(spec, "Issue Resolution (other)")

        decision = classify_and_route(
            benchmark="swebenchmultimodal",
            instance_data=self._image_instance(),
            router=spec,
        )

        assert decision.forced_vision_fallback is True
        assert decision.chosen_model_id in {"gpt-5.5", "kimi-k2.6"}

    def test_vision_tier_keeps_assignment(self) -> None:
        spec = _router_spec()
        _attach_stub(spec, "Frontend")

        decision = classify_and_route(
            benchmark="swebenchmultimodal",
            instance_data=self._image_instance(),
            router=spec,
        )

        # Frontend → kimi, which is vision-capable; no redirect needed.
        assert decision.forced_vision_fallback is False
        assert decision.chosen_model_id == "kimi-k2.6"

    def test_text_only_instance_unaffected(self) -> None:
        spec = _router_spec()
        _attach_stub(spec, "Issue Resolution (other)")

        decision = classify_and_route(
            benchmark="swebench",
            instance_data={"problem_statement": "Fix the pagination off-by-one"},
            router=spec,
        )

        assert decision.forced_vision_fallback is False
        assert decision.chosen_model_id == "minimax-m2.7"

    def test_image_assets_as_json_string(self) -> None:
        spec = _router_spec()
        _attach_stub(spec, "Issue Resolution (other)")

        instance = {
            "problem_statement": "Bug.",
            "image_assets": json.dumps(
                {"problem_statement": ["https://example.com/a.png"]}
            ),
        }

        decision = classify_and_route(
            benchmark="swebenchmultimodal",
            instance_data=instance,
            router=spec,
        )

        assert decision.forced_vision_fallback is True
