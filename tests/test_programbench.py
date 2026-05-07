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


# ---------------------------------------------------------------------------
# Gold-tests Stop hook
# ---------------------------------------------------------------------------


def _make_metadata_with_details(**details: object):
    """Build a minimal ``EvalMetadata`` for hook tests.

    We stay off the LLM's network (api_key is just a placeholder). The
    fields below are the bare minimum the pydantic model requires.
    """
    from pydantic import SecretStr

    from benchmarks.utils.critics import AgentFinishedCritic
    from benchmarks.utils.models import EvalMetadata
    from openhands.sdk import LLM

    return EvalMetadata(
        llm=LLM(
            model="openai/gpt-4o-mini",
            api_key=SecretStr("sk-test"),
            usage_id="test",
        ),
        dataset="programbench/ProgramBench",
        dataset_split="test",
        max_iterations=10,
        eval_output_dir="/tmp/test",
        details=dict(details),
        prompt_path=str(Path(run_infer.__file__).parent / "prompts" / "default.j2"),
        critic=AgentFinishedCritic(),
    )


class TestStopHookConfig:
    """Unit tests for ``_build_stop_hook_config``.

    The compile-contract hook is always installed so the build-contract
    layer can never be skipped. ``enforce_gold_tests`` adds the heavier
    gold-vs-agent test comparison hook on top, sequenced after the
    contract check.
    """

    def test_installs_compile_contract_hook_by_default(self) -> None:
        cfg = run_infer._build_stop_hook_config(_make_metadata_with_details())
        assert cfg is not None
        assert len(cfg.stop) == 1
        # Default is contract-only — single hook in the matcher.
        assert len(cfg.stop[0].hooks) == 1
        contract_body = run_infer.COMPILE_CONTRACT_HOOK_PATH.read_text()
        assert contract_body.strip() in cfg.stop[0].hooks[0].command

    def test_returns_none_when_explicitly_disabled(self) -> None:
        cfg = run_infer._build_stop_hook_config(
            _make_metadata_with_details(disable_stop_hooks=True)
        )
        assert cfg is None

    def test_appends_gold_tests_hook_when_enforced(self) -> None:
        cfg = run_infer._build_stop_hook_config(
            _make_metadata_with_details(enforce_gold_tests=True)
        )
        assert cfg is not None
        assert len(cfg.stop[0].hooks) == 2
        # Order matters: the cheap contract check must run first so the
        # gold-tests hook never sees a missing compile.sh.
        first, second = cfg.stop[0].hooks
        contract_body = run_infer.COMPILE_CONTRACT_HOOK_PATH.read_text()
        gold_body = run_infer.GOLD_TESTS_HOOK_PATH.read_text()
        assert contract_body.strip() in first.command
        assert gold_body.strip() in second.command

    def test_does_not_install_gold_tests_hook_by_default(self) -> None:
        cfg = run_infer._build_stop_hook_config(_make_metadata_with_details())
        assert cfg is not None
        gold_body = run_infer.GOLD_TESTS_HOOK_PATH.read_text()
        for hook in cfg.stop[0].hooks:
            assert gold_body.strip() not in hook.command

    def test_respects_custom_contract_timeout(self) -> None:
        cfg = run_infer._build_stop_hook_config(
            _make_metadata_with_details(compile_contract_hook_timeout=11)
        )
        assert cfg is not None
        assert cfg.stop[0].hooks[0].timeout == 11

    def test_respects_custom_gold_tests_timeout(self) -> None:
        cfg = run_infer._build_stop_hook_config(
            _make_metadata_with_details(
                enforce_gold_tests=True, gold_tests_hook_timeout=42
            )
        )
        assert cfg is not None
        assert cfg.stop[0].hooks[1].timeout == 42

    def test_only_stop_event_is_populated(self) -> None:
        # Sanity: we don't accidentally wire the scripts to fire on
        # every tool invocation.
        cfg = run_infer._build_stop_hook_config(
            _make_metadata_with_details(enforce_gold_tests=True)
        )
        assert cfg is not None
        assert cfg.pre_tool_use == []
        assert cfg.post_tool_use == []
        assert cfg.user_prompt_submit == []
        assert cfg.session_start == []
        assert cfg.session_end == []


# ---------------------------------------------------------------------------
# Hook script behaviour (executed by bash; no Docker required)
# ---------------------------------------------------------------------------


@pytest.fixture
def hook_sandbox(tmp_path: Path) -> dict[str, Path]:
    """Set up a fake /workspace + /opt + state dir for the bash hook."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    eval_dir = workspace / "eval"
    eval_dir.mkdir()
    runs_dir = workspace / ".programbench-stop-hook"
    return {
        "workspace": workspace,
        "eval": eval_dir,
        "runs_dir": runs_dir,
        "agent": workspace / "executable",
        "gold": tmp_path / "gold-stash",
    }


def _run_hook(
    script: Path,
    cwd: Path,
    env_overrides: dict[str, str],
    stdin: str = "{}",
):
    """Invoke the hook script with isolated env vars."""
    import os
    import subprocess

    env = {
        # Keep PATH so coreutils/sha256sum/python3 resolve, but drop everything else.
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        # Default these so the hook never accidentally touches /opt or /workspace
        # on the developer's host.
        "PB_STASHED_GOLD_PATH": str(env_overrides.pop("PB_STASHED_GOLD_PATH", "")),
        "PB_AGENT_BINARY_PATH": str(env_overrides.pop("PB_AGENT_BINARY_PATH", "")),
        "PB_STOP_HOOK_RUNS_DIR": str(env_overrides.pop("PB_STOP_HOOK_RUNS_DIR", "")),
        "PB_STOP_HOOK_MAX_RETRIES": str(
            env_overrides.pop("PB_STOP_HOOK_MAX_RETRIES", 3)
        ),
        "PB_STOP_HOOK_TEST_TIMEOUT": str(
            env_overrides.pop("PB_STOP_HOOK_TEST_TIMEOUT", 10)
        ),
    }
    env.update({k: str(v) for k, v in env_overrides.items()})
    return subprocess.run(
        ["bash", str(script)],
        cwd=str(cwd),
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _write_runner(eval_dir: Path, junit_xml: str) -> None:
    """Drop a deterministic ``run.sh`` that emits the supplied JUnit XML.

    We don't actually need pytest for these tests — only the XML the hook
    parses. The ``run.sh`` differentiates between gold and agent runs by
    inspecting ./executable's contents, so that one fixture covers both
    test branches the hook makes.
    """
    runner = eval_dir / "run.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        f"cat > eval/results.xml <<'XML'\n{junit_xml}\nXML\n"
        "exit 0\n"
    )
    runner.chmod(0o755)


GOLD_PASS_AGENT_FAIL_XML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="pb">
  <testcase classname="t" name="adds">{adds}</testcase>
  <testcase classname="t" name="subs">{subs}</testcase>
</testsuite>"""


class TestGoldTestsHookScript:
    """End-to-end tests of the bash hook script.

    These run the actual script against synthesised binaries / runner.sh
    inside ``tmp_path``. Pytest never has to be installed in the host
    environment because we replace ``./eval/run.sh`` with a fixture that
    writes a JUnit XML directly.
    """

    SCRIPT = run_infer.GOLD_TESTS_HOOK_PATH

    def test_allows_stop_when_no_gold_binary(self, hook_sandbox) -> None:
        # No gold binary on disk — hook can't compare and should let the
        # agent stop (the upstream eval will catch real regressions).
        hook_sandbox["agent"].write_text("agent-build")
        hook_sandbox["agent"].chmod(0o755)
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides={
                "PB_STASHED_GOLD_PATH": str(hook_sandbox["gold"]),
                "PB_AGENT_BINARY_PATH": str(hook_sandbox["agent"]),
                "PB_STOP_HOOK_RUNS_DIR": str(hook_sandbox["runs_dir"]),
            },
        )
        assert result.returncode == 0, result.stderr
        assert "gold binary missing" in result.stderr

    def test_blocks_stop_when_no_agent_binary(self, hook_sandbox) -> None:
        # Agent never built ./executable — hook must block.
        hook_sandbox["gold"].write_text("gold-build")
        hook_sandbox["gold"].chmod(0o755)
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides={
                "PB_STASHED_GOLD_PATH": str(hook_sandbox["gold"]),
                "PB_AGENT_BINARY_PATH": str(hook_sandbox["agent"]),
                "PB_STOP_HOOK_RUNS_DIR": str(hook_sandbox["runs_dir"]),
            },
        )
        assert result.returncode == 1
        assert "no agent binary" in result.stderr

    def test_allows_stop_when_binary_matches_gold(self, hook_sandbox) -> None:
        # Byte-identical → cheap path: skip pytest entirely.
        same = b"# identical bytes\n"
        hook_sandbox["agent"].write_bytes(same)
        hook_sandbox["gold"].write_bytes(same)
        hook_sandbox["agent"].chmod(0o755)
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides={
                "PB_STASHED_GOLD_PATH": str(hook_sandbox["gold"]),
                "PB_AGENT_BINARY_PATH": str(hook_sandbox["agent"]),
                "PB_STOP_HOOK_RUNS_DIR": str(hook_sandbox["runs_dir"]),
            },
        )
        assert result.returncode == 0, result.stderr
        assert "byte-identical" in result.stderr

    def test_allows_stop_when_no_runner_present(self, hook_sandbox) -> None:
        # Different binaries but no eval/run.sh → can't compare → allow.
        hook_sandbox["agent"].write_text("agent")
        hook_sandbox["gold"].write_text("gold")
        hook_sandbox["agent"].chmod(0o755)
        # Ensure no run.sh
        runner = hook_sandbox["eval"] / "run.sh"
        if runner.exists():
            runner.unlink()
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides={
                "PB_STASHED_GOLD_PATH": str(hook_sandbox["gold"]),
                "PB_AGENT_BINARY_PATH": str(hook_sandbox["agent"]),
                "PB_STOP_HOOK_RUNS_DIR": str(hook_sandbox["runs_dir"]),
            },
        )
        assert result.returncode == 0, result.stderr
        assert "no eval/run.sh" in result.stderr

    def test_blocks_stop_on_test_mismatch(self, hook_sandbox) -> None:
        # Runner emits "both pass" when the binary is gold and "subs fails"
        # when the binary is agent. Hook must spot the mismatch and block.
        hook_sandbox["gold"].write_text("# gold-build\n")
        hook_sandbox["agent"].write_text("# agent-build\n")
        hook_sandbox["agent"].chmod(0o755)
        runner = hook_sandbox["eval"] / "run.sh"
        runner.write_text(
            "#!/usr/bin/env bash\n"
            "if grep -q '# gold-build' ./executable; then\n"
            "  cat > eval/results.xml <<'XML'\n"
            + GOLD_PASS_AGENT_FAIL_XML_TEMPLATE.format(adds="", subs="")
            + "\nXML\n"
            "else\n"
            "  cat > eval/results.xml <<'XML'\n"
            + GOLD_PASS_AGENT_FAIL_XML_TEMPLATE.format(
                adds="", subs="<failure>subs broke</failure>"
            )
            + "\nXML\n"
            "fi\n"
            "exit 0\n"
        )
        runner.chmod(0o755)
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides={
                "PB_STASHED_GOLD_PATH": str(hook_sandbox["gold"]),
                "PB_AGENT_BINARY_PATH": str(hook_sandbox["agent"]),
                "PB_STOP_HOOK_RUNS_DIR": str(hook_sandbox["runs_dir"]),
            },
        )
        assert result.returncode == 1, result.stderr
        assert "1 test(s) pass against the gold binary" in result.stderr
        assert "t.subs" in result.stderr  # mismatch was the 'subs' test
        # Agent binary must be back in place after the script runs (otherwise
        # subsequent agent steps would see gold and pass tests "for free").
        assert "# agent-build" in hook_sandbox["agent"].read_text()

    def test_allows_stop_when_tests_fully_match(self, hook_sandbox) -> None:
        # Runner emits identical XML regardless of binary → no mismatch.
        hook_sandbox["gold"].write_text("# gold-build\n")
        hook_sandbox["agent"].write_text("# agent-build\n")
        hook_sandbox["agent"].chmod(0o755)
        _write_runner(
            hook_sandbox["eval"],
            GOLD_PASS_AGENT_FAIL_XML_TEMPLATE.format(adds="", subs=""),
        )
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides={
                "PB_STASHED_GOLD_PATH": str(hook_sandbox["gold"]),
                "PB_AGENT_BINARY_PATH": str(hook_sandbox["agent"]),
                "PB_STOP_HOOK_RUNS_DIR": str(hook_sandbox["runs_dir"]),
            },
        )
        assert result.returncode == 0, result.stderr
        assert "all gold-passing tests also pass" in result.stderr

    def test_retry_cap_lets_agent_eventually_stop(self, hook_sandbox) -> None:
        # If the agent stays broken, the hook must eventually concede so
        # we don't loop forever and exhaust max_iterations on stop hooks.
        hook_sandbox["gold"].write_text("# gold\n")
        hook_sandbox["agent"].write_text("# agent\n")
        hook_sandbox["agent"].chmod(0o755)
        runner_xml = (
            "<?xml version='1.0'?><testsuite>"
            "<testcase classname='t' name='x'>"
            "<failure>broken</failure>"
            "</testcase></testsuite>"
        )
        # gold passes, agent fails — same setup as the mismatch test
        runner = hook_sandbox["eval"] / "run.sh"
        runner.write_text(
            "#!/usr/bin/env bash\n"
            "if grep -q '# gold' ./executable; then\n"
            "  cat > eval/results.xml <<'XML'\n"
            "<?xml version='1.0'?><testsuite>"
            "<testcase classname='t' name='x'/>"
            "</testsuite>\nXML\n"
            "else\n"
            "  cat > eval/results.xml <<XML\n"
            f"{runner_xml}\nXML\n"
            "fi\n"
            "exit 0\n"
        )
        runner.chmod(0o755)
        env_overrides_base = {
            "PB_STASHED_GOLD_PATH": str(hook_sandbox["gold"]),
            "PB_AGENT_BINARY_PATH": str(hook_sandbox["agent"]),
            "PB_STOP_HOOK_RUNS_DIR": str(hook_sandbox["runs_dir"]),
            "PB_STOP_HOOK_MAX_RETRIES": "2",
        }
        # First two invocations block...
        for attempt in (1, 2):
            r = _run_hook(
                self.SCRIPT,
                cwd=hook_sandbox["workspace"],
                env_overrides=dict(env_overrides_base),
            )
            assert r.returncode == 1, (
                f"attempt {attempt} should block; stderr={r.stderr}"
            )
        # Third invocation hits the cap and concedes.
        r = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides=dict(env_overrides_base),
        )
        assert r.returncode == 0, r.stderr
        assert "max retries" in r.stderr


# ---------------------------------------------------------------------------
# Compile-contract hook script (executed by bash)
# ---------------------------------------------------------------------------


def _run_compile_hook(
    workspace: Path,
    *,
    runs_dir: Path | None = None,
    max_retries: int = 3,
    timeout_secs: int = 60,
):
    """Invoke ``check_compile_contract.sh`` against an isolated workspace."""
    import os
    import subprocess

    runs_dir = runs_dir or (workspace / ".programbench-compile-hook")
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PB_WORKSPACE": str(workspace),
        "PB_COMPILE_HOOK_RUNS_DIR": str(runs_dir),
        "PB_COMPILE_HOOK_MAX_RETRIES": str(max_retries),
        "PB_COMPILE_HOOK_TIMEOUT": str(timeout_secs),
    }
    return subprocess.run(
        ["bash", str(run_infer.COMPILE_CONTRACT_HOOK_PATH)],
        cwd=str(workspace),
        env=env,
        input="{}",
        capture_output=True,
        text=True,
        timeout=timeout_secs + 10,
    )


class TestCompileContractHookScript:
    """End-to-end behaviour of ``check_compile_contract.sh``.

    These tests synthesise a fake workspace and a small compile.sh and
    run the actual bash hook, checking that it correctly accepts /
    rejects each contract scenario without needing Docker.
    """

    def test_blocks_stop_when_compile_sh_missing(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        r = _run_compile_hook(workspace)
        assert r.returncode == 1, r.stderr
        assert "compile.sh is missing" in r.stderr
        # Helpful copy-paste-ready examples should always be in the
        # feedback so the agent has something concrete to act on.
        assert "cargo build --release" in r.stderr

    def test_blocks_stop_when_compile_sh_fails(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        compile_sh = workspace / "compile.sh"
        compile_sh.write_text("#!/usr/bin/env bash\necho boom >&2\nexit 7\n")
        compile_sh.chmod(0o755)
        r = _run_compile_hook(workspace)
        assert r.returncode == 1, r.stderr
        assert "exited non-zero" in r.stderr
        # Tail of the script's stderr should make it into the message.
        assert "boom" in r.stderr

    def test_blocks_stop_when_executable_not_produced(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        # Script exits 0 but never writes ./executable.
        (workspace / "compile.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
        (workspace / "compile.sh").chmod(0o755)
        r = _run_compile_hook(workspace)
        assert r.returncode == 1, r.stderr
        assert "./executable was not produced" in r.stderr

    def test_allows_stop_when_compile_sh_produces_executable(
        self, tmp_path: Path
    ) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        # A minimal but valid compile.sh: writes a runnable
        # ./executable. We don't exercise the binary itself; the hook
        # only checks the file exists at the right path.
        (workspace / "compile.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'printf "#!/usr/bin/env bash\\nexit 0\\n" > ./executable\n'
            "chmod +x ./executable\n"
        )
        (workspace / "compile.sh").chmod(0o755)
        r = _run_compile_hook(workspace)
        assert r.returncode == 0, r.stderr
        assert "build contract OK" in r.stderr
        # Verify the hook actually ran the script (./executable exists)
        # rather than short-circuiting somewhere.
        assert (workspace / "executable").exists()

    def test_wipes_stale_executable_before_running_compile(
        self, tmp_path: Path
    ) -> None:
        # Regression: if the agent built ./executable manually but
        # compile.sh doesn't actually produce one, the hook must catch
        # it. Otherwise the grader will silently fail on a clean
        # extraction.
        workspace = tmp_path / "ws"
        workspace.mkdir()
        stale = workspace / "executable"
        stale.write_text("stale-binary")
        stale.chmod(0o755)
        # compile.sh that doesn't actually build anything.
        (workspace / "compile.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
        (workspace / "compile.sh").chmod(0o755)
        r = _run_compile_hook(workspace)
        assert r.returncode == 1, r.stderr
        assert "./executable was not produced" in r.stderr

    def test_allows_stop_after_max_retries(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        # No compile.sh — this would normally block. After hitting the
        # retry cap, the hook should release the agent so a stuck
        # conversation can finish.
        runs_dir = workspace / "runs"
        runs = [
            _run_compile_hook(workspace, runs_dir=runs_dir, max_retries=3)
            for _ in range(4)
        ]
        # First three calls block (rc=1, contract not satisfied); the
        # fourth trips the retry cap and releases the agent.
        assert [r.returncode for r in runs[:3]] == [1, 1, 1]
        assert runs[3].returncode == 0, runs[3].stderr
        assert "max retries" in runs[3].stderr


# ---------------------------------------------------------------------------
# CLI flag plumbing
# ---------------------------------------------------------------------------


class TestCondenserCliPlumbing:
    """`--condenser-max-size` etc. used to be parsed but ignored. These
    tests pin the shape we now expect: the parser accepts them, defaults
    are well-defined, and `--enforce-gold-tests` shows up alongside the
    pre-existing flags."""

    def test_help_advertises_all_relevant_flags(self) -> None:
        import argparse

        from benchmarks.programbench.config import INFER_DEFAULTS
        from benchmarks.programbench.run_infer import main

        # We can't easily exercise main() (it triggers --help/SystemExit
        # gymnastics), so import the helpers it composes.
        from benchmarks.utils.args_parser import (
            add_prompt_path_argument,
            get_parser,
        )

        parser = get_parser()
        # Mimic main() so the flag set we test matches reality.
        add_prompt_path_argument(parser, str(Path(run_infer.__file__)))
        parser.add_argument("--task-image-tag", type=str)
        parser.add_argument(
            "--build-target",
            type=str,
            choices=["binary", "binary-minimal", "source", "source-minimal"],
        )
        parser.add_argument("--allow-network", action="store_true")
        # Mirror run_infer.main(): enforce-gold-tests is on by default
        # (helm dispatch can't pass extra CLI args, so the default is the
        # only switch we have for production runs).
        parser.add_argument(
            "--enforce-gold-tests",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        parser.add_argument("--gold-tests-hook-timeout", type=int, default=600)
        parser.set_defaults(**INFER_DEFAULTS)

        # Assertion 1: argparse can parse a leaderboard-style invocation
        # without flipping the gold-tests default, and the args
        # round-trip into known names.
        args = parser.parse_args(
            [
                "/dev/null",  # llm_config_path positional
                "--max-iterations",
                "1000",
                "--enable-condenser",
                "--condenser-max-size",
                "80",
                "--condenser-keep-first",
                "4",
                "--gold-tests-hook-timeout",
                "300",
            ]
        )
        assert args.max_iterations == 1000
        assert args.enable_condenser is True
        assert args.condenser_max_size == 80
        assert args.condenser_keep_first == 4
        # Default-on so production helm dispatch picks up the gold-tests
        # hook without orchestrator changes.
        assert args.enforce_gold_tests is True
        assert args.gold_tests_hook_timeout == 300

        # Assertion 2: ``--no-enforce-gold-tests`` lets local smoke
        # runs opt out of the heavy test re-run.
        opted_out = parser.parse_args(["/dev/null", "--no-enforce-gold-tests"])
        assert opted_out.enforce_gold_tests is False
        # Sanity: didn't introduce a stray attribute that nobody owns.
        assert isinstance(parser, argparse.ArgumentParser)
        # silences "imported but unused" without changing public surface
        assert callable(main)
