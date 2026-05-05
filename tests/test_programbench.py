"""Tests for the ProgramBench benchmark module.

These tests exercise the parts of the ProgramBench integration that don't
require a Docker daemon or the real upstream task images: instance image
naming, selection logic, prompt rendering, eval-result aggregation, and the
``run_dir`` resolver used by ``programbench-eval``.

We deliberately avoid mocking the agent / workspace pipeline — those paths
are exercised end-to-end by the CI smoke workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.programbench import run_infer
from benchmarks.programbench.eval_infer import (
    aggregate_eval_results,
    get_run_dir,
)


# ---------------------------------------------------------------------------
# run_infer helpers
# ---------------------------------------------------------------------------


class TestInstanceToImage:
    def test_double_underscore_replaced_with_1776(self) -> None:
        # ProgramBench's image-naming convention is documented in their docs:
        # repo separator '__' becomes '_1776_' so Docker accepts the tag.
        image = run_infer._instance_to_image(
            "abishekvashok__cmatrix.5c082c6", "task_cleanroom"
        )
        assert image == "programbench/abishekvashok_1776_cmatrix.5c082c6:task_cleanroom"

    def test_uses_provided_tag(self) -> None:
        image = run_infer._instance_to_image("foo__bar.deadbee", "task")
        assert image.endswith(":task")

    def test_handles_no_double_underscore(self) -> None:
        # Defensive: if a future instance id doesn't have '__', we should
        # still produce a syntactically valid image reference rather than
        # silently mangle it.
        image = run_infer._instance_to_image("solo.deadbee", "task_cleanroom")
        assert image == "programbench/solo.deadbee:task_cleanroom"


class TestSelectInstances:
    def _instances(self) -> list[dict]:
        return [
            {"instance_id": "alpha__a.000"},
            {"instance_id": "beta__b.111"},
            {"instance_id": "gamma__c.222"},
        ]

    def test_returns_all_when_no_filters(self) -> None:
        out = run_infer._select_instances(self._instances(), None, 0)
        assert [i["instance_id"] for i in out] == [
            "alpha__a.000",
            "beta__b.111",
            "gamma__c.222",
        ]

    def test_n_limit_truncates(self) -> None:
        out = run_infer._select_instances(self._instances(), None, 2)
        assert len(out) == 2

    def test_select_file_filters_to_subset(self, tmp_path: Path) -> None:
        select = tmp_path / "select.txt"
        select.write_text("alpha__a.000\ngamma__c.222\n")
        out = run_infer._select_instances(self._instances(), str(select), 0)
        assert {i["instance_id"] for i in out} == {
            "alpha__a.000",
            "gamma__c.222",
        }

    def test_select_file_with_blank_lines_and_trailing_whitespace(
        self, tmp_path: Path
    ) -> None:
        select = tmp_path / "select.txt"
        select.write_text("\nalpha__a.000\n\n  beta__b.111  \n\n")
        out = run_infer._select_instances(self._instances(), str(select), 0)
        assert {i["instance_id"] for i in out} == {"alpha__a.000", "beta__b.111"}

    def test_select_file_unknown_id_raises(self, tmp_path: Path) -> None:
        select = tmp_path / "select.txt"
        select.write_text("alpha__a.000\ndoesnotexist\n")
        with pytest.raises(ValueError, match="unknown instance ids"):
            run_infer._select_instances(self._instances(), str(select), 0)

    def test_empty_select_file_raises(self, tmp_path: Path) -> None:
        select = tmp_path / "select.txt"
        select.write_text("\n\n   \n")
        with pytest.raises(ValueError, match="empty"):
            run_infer._select_instances(self._instances(), str(select), 0)


class TestRenderInstruction:
    def test_renders_default_template(self, tmp_path: Path) -> None:
        # Use the actual default.j2 template so we catch breakage if the
        # template grows new variables that the renderer doesn't supply.
        from benchmarks.utils.models import EvalMetadata
        from openhands.sdk import LLM
        from openhands.sdk.critic import PassCritic

        prompt_path = (
            Path(run_infer.__file__).parent / "prompts" / "default.j2"
        ).resolve()

        metadata = EvalMetadata(
            llm=LLM(model="dummy", usage_id="test"),
            dataset="programbench/ProgramBench",
            max_iterations=10,
            eval_output_dir=str(tmp_path),
            prompt_path=str(prompt_path),
            critic=PassCritic(),
        )
        instance = {
            "instance_id": "abishekvashok__cmatrix.5c082c6",
            "repository": "abishekvashok/cmatrix",
            "language": "c",
        }
        instruction = run_infer._render_instruction(instance, metadata)
        # Sanity: the template must drop key facts about the task in.
        assert "/workspace" in instruction
        assert "/workspace/cmatrix" in instruction  # binary path hint
        assert "abishekvashok/cmatrix" in instruction
        # The default template formats the language hint as `c` (backticked).
        assert "`c`" in instruction
        # Negative: the prompt MUST tell the agent it has no internet — this
        # is the load-bearing ProgramBench invariant.
        assert "no internet" in instruction.lower()


# ---------------------------------------------------------------------------
# eval_infer aggregation
# ---------------------------------------------------------------------------


def _write_eval_json(run_dir: Path, instance_id: str, payload: dict) -> Path:
    """Write a synthetic ``<id>/<id>.eval.json`` and (empty) submission."""
    inst_dir = run_dir / instance_id
    inst_dir.mkdir(parents=True, exist_ok=True)
    (inst_dir / "submission.tar.gz").write_bytes(b"")
    eval_path = inst_dir / f"{instance_id}.eval.json"
    eval_path.write_text(json.dumps(payload))
    return eval_path


class TestAggregateEvalResults:
    def test_resolved_when_all_tests_pass_and_no_error(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_eval_json(
            run_dir,
            "alpha__a.000",
            {
                "test_results": [
                    {"name": "t1", "branch": "b", "status": "passed", "extra": {}},
                    {"name": "t2", "branch": "b", "status": "passed", "extra": {}},
                ],
                "error_code": None,
            },
        )
        report = aggregate_eval_results(run_dir, ["alpha__a.000"])
        assert report["resolved_instances"] == 1
        assert report["unresolved_instances"] == 0
        assert report["error_instances"] == 0
        assert report["resolved_ids"] == ["alpha__a.000"]

    def test_almost_resolved_threshold(self, tmp_path: Path) -> None:
        # 19/20 = 95% → counts as almost-resolved; 18/20 = 90% → does not.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_eval_json(
            run_dir,
            "almost__a.000",
            {
                "test_results": [
                    {
                        "name": f"t{i}",
                        "branch": "b",
                        "status": "passed" if i > 0 else "failure",
                        "extra": {},
                    }
                    for i in range(20)
                ],
                "error_code": None,
            },
        )
        _write_eval_json(
            run_dir,
            "below__b.000",
            {
                "test_results": [
                    {
                        "name": f"t{i}",
                        "branch": "b",
                        "status": "passed" if i >= 2 else "failure",
                        "extra": {},
                    }
                    for i in range(20)
                ],
                "error_code": None,
            },
        )
        report = aggregate_eval_results(run_dir, ["almost__a.000", "below__b.000"])
        assert report["almost_resolved_ids"] == ["almost__a.000"]
        assert report["unresolved_instances"] == 2
        assert report["resolved_instances"] == 0

    def test_error_code_classifies_as_error(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_eval_json(
            run_dir,
            "broken__b.000",
            {
                "test_results": [],
                "error_code": "build_failed",
                "error_details": "compilation crashed",
            },
        )
        report = aggregate_eval_results(run_dir, ["broken__b.000"])
        assert report["error_instances"] == 1
        assert report["resolved_instances"] == 0
        assert "broken__b.000" in report["error_ids"]
        assert "broken__b.000" in report["unresolved_ids"]

    def test_missing_eval_json_is_incomplete(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "missing__m.000").mkdir(parents=True)
        report = aggregate_eval_results(run_dir, ["missing__m.000"])
        assert report["completed_instances"] == 0
        assert report["incomplete_instances"] == 1
        assert "missing__m.000" in report["error_ids"]

    def test_malformed_eval_json_is_error(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        inst_dir = run_dir / "bad__json.000"
        inst_dir.mkdir(parents=True)
        (inst_dir / "bad__json.000.eval.json").write_text("{not json")
        report = aggregate_eval_results(run_dir, ["bad__json.000"])
        assert report["error_instances"] == 1

    def test_empty_test_results_with_no_error_is_unresolved(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_eval_json(
            run_dir,
            "empty__e.000",
            {"test_results": [], "error_code": None},
        )
        report = aggregate_eval_results(run_dir, ["empty__e.000"])
        # Eval ran cleanly but no tests fired → unresolved (not error).
        # This matches the upstream interpretation: a zero-test run is a
        # zero-score, not a harness failure.
        assert report["completed_instances"] == 1
        assert report["unresolved_instances"] == 1
        assert report["resolved_instances"] == 0
        assert report["error_instances"] == 0

    def test_mixed_run(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_eval_json(
            run_dir,
            "good__g.000",
            {
                "test_results": [
                    {"name": "t", "branch": "b", "status": "passed", "extra": {}}
                ],
                "error_code": None,
            },
        )
        _write_eval_json(
            run_dir,
            "fail__f.000",
            {
                "test_results": [
                    {"name": "t", "branch": "b", "status": "failure", "extra": {}}
                ],
                "error_code": None,
            },
        )
        _write_eval_json(
            run_dir,
            "err__e.000",
            {"test_results": [], "error_code": "boom"},
        )
        report = aggregate_eval_results(
            run_dir, ["good__g.000", "fail__f.000", "err__e.000"]
        )
        assert report["resolved_instances"] == 1
        assert report["unresolved_instances"] == 2
        assert report["error_instances"] == 1
        assert report["total_instances"] == 3
        assert report["submitted_ids"] == [
            "err__e.000",
            "fail__f.000",
            "good__g.000",
        ]


class TestResolveRunDir:
    def test_finds_sibling_run_directory(self, tmp_path: Path) -> None:
        eval_dir = tmp_path / "eval_outputs" / "model_sdk_X_maxiter_200"
        eval_dir.mkdir(parents=True)
        (eval_dir / "run").mkdir()
        output_jsonl = eval_dir / "output.jsonl"
        output_jsonl.write_text("")
        assert get_run_dir(output_jsonl) == eval_dir / "run"

    def test_missing_run_dir_raises(self, tmp_path: Path) -> None:
        (tmp_path / "output.jsonl").write_text("")
        with pytest.raises(FileNotFoundError, match="ProgramBench submissions"):
            get_run_dir(tmp_path / "output.jsonl")
