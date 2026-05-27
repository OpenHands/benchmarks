"""Tests for the ProgramBench benchmark module.

These tests exercise the parts of the ProgramBench integration that don't
require a Docker daemon or the real upstream task images: instance image
naming, selection logic, prompt rendering, eval-result aggregation, and the
``run_dir`` resolver used by ``programbench-eval``.

We deliberately avoid mocking the agent / workspace pipeline — those paths
are exercised end-to-end by the CI smoke workflow.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from benchmarks.programbench import run_infer
from benchmarks.programbench.eval_infer import (
    _run_programbench_eval,
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
        # Binary path: post-Step-0 stable name, NOT a per-instance hint.
        # Pre-retry-21 the prompt rendered ``/workspace/<repo_name>`` here,
        # but that path doesn't exist in the cleanroom image (the actual
        # reference is at ``/workspace/executable``, mode ---x--x--x);
        # Step 0 of the prompt now instructs the agent to ``mv`` it to
        # ``/workspace/executable.ref`` before doing anything else.
        assert "/workspace/executable.ref" in instruction
        assert "Step 0" in instruction
        # Per-instance metadata still surfaces (just not as the binary path):
        assert "abishekvashok/cmatrix" in instruction
        # The default template formats the language hint as `c` (backticked).
        assert "`c`" in instruction
        # Negative: the prompt MUST tell the agent it has no internet — this
        # is the load-bearing ProgramBench invariant.
        assert "no internet" in instruction.lower()


# ---------------------------------------------------------------------------
# Submission tarball shape (workspace-isolation contract)
# ---------------------------------------------------------------------------


class TestSubmissionTarballShape:
    """Pin the ``tar`` invocation produced by ``_collect_submission``.

    The agent-server flushes events into ``/workspace/conversations/`` and
    ``/workspace/bash_events/`` asynchronously — even after
    ``conversation.run()`` returns — so without explicit excludes tar
    races those writes and aborts with ``tar: .: file changed as we
    read it``. These tests pin the defences so we don't silently
    regress them and re-introduce the retry-13 failure mode."""

    def _make_workspace(self):  # -> tuple[RemoteWorkspace-shaped Mock, list[str]]
        captured: list[str] = []

        def fake_execute_command(cmd: str, timeout: int = 0):
            captured.append(cmd)
            r = MagicMock()
            r.exit_code = 0
            r.stdout = ""
            r.stderr = ""
            return r

        def fake_file_download(src: str, dst: str):
            Path(dst).write_bytes(b"fake-archive")
            r = MagicMock()
            r.success = True
            return r

        workspace = MagicMock()
        workspace.execute_command = fake_execute_command
        workspace.file_download = fake_file_download
        # Force the download_directory branch to fall through; otherwise
        # MagicMock auto-creates a truthy attribute.
        workspace.download_directory = None
        return workspace, captured

    def _make_evaluation(self, tmp_path: Path):
        from benchmarks.programbench.run_infer import ProgramBenchEvaluation
        from benchmarks.utils.models import EvalMetadata
        from openhands.sdk import LLM
        from openhands.sdk.critic import PassCritic

        prompt_path = (
            Path(run_infer.__file__).parent / "prompts" / "default.j2"
        ).resolve()
        return ProgramBenchEvaluation(
            metadata=EvalMetadata(
                llm=LLM(model="dummy", usage_id="test"),
                dataset="programbench/ProgramBench",
                max_iterations=10,
                eval_output_dir=str(tmp_path),
                prompt_path=str(prompt_path),
                critic=PassCritic(),
            ),
        )

    def test_excludes_async_state_dirs_and_tolerates_warning(
        self, tmp_path: Path
    ) -> None:
        from benchmarks.utils.models import EvalInstance

        evaluation = self._make_evaluation(tmp_path)
        workspace, captured = self._make_workspace()
        instance = EvalInstance(
            id="abishekvashok__cmatrix.5c082c6",
            data={
                "repository": "abishekvashok/cmatrix",
                "task_image": "programbench/cmatrix:task_cleanroom",
            },
        )

        # The duck-typed workspace mock is fine at runtime; cast away
        # the strict RemoteWorkspace type for pyright.
        from openhands.sdk.workspace.remote import RemoteWorkspace

        evaluation._collect_submission(instance, cast(RemoteWorkspace, workspace))

        assert len(captured) == 1, (
            f"expected exactly one tar invocation, captured {len(captured)}"
        )
        tar_cmd = captured[0]
        # Defences against the agent-server's async event flush — these
        # are the actual root cause of the retry-13 tar race.
        assert "--warning=no-file-changed" in tar_cmd, (
            "tar must tolerate the 'file changed as we read it' warning "
            "from concurrent agent-server writes; otherwise the orchestrator "
            "fails the instance even though the archive is intact."
        )
        assert "--exclude=./conversations" in tar_cmd, (
            "agent-server flushes event journals to /workspace/conversations/; "
            "they must be excluded so tar isn't racing them."
        )
        assert "--exclude=./bash_events" in tar_cmd, (
            "agent-server flushes bash command history to "
            "/workspace/bash_events/; they must be excluded so tar isn't "
            "racing them."
        )
        # Defences against unreadable files (root-owned reference
        # binaries and any *.orig copies the agent may have made).
        assert "--ignore-failed-read" in tar_cmd, (
            "tar must skip rather than abort when any /workspace file is "
            "permission-denied; the eval harness ignores stray files anyway."
        )
        assert "--exclude=./executable" in tar_cmd
        assert "--exclude=./executable.orig" in tar_cmd, (
            "agents commonly do ``cp executable executable.orig`` to "
            "preserve the reference binary; cp preserves perms so the "
            "copy is also root-owned 0700 and trips tar."
        )
        assert "cmatrix" in tar_cmd  # repo basename via shlex.quote

    def test_uses_file_download_without_base64_fallback(self, tmp_path: Path) -> None:
        from benchmarks.utils.models import EvalInstance
        from openhands.sdk.workspace.remote import RemoteWorkspace

        evaluation = self._make_evaluation(tmp_path)
        workspace, captured = self._make_workspace()
        instance = EvalInstance(
            id="abishekvashok__cmatrix.5c082c6",
            data={
                "repository": "abishekvashok/cmatrix",
                "task_image": "programbench/cmatrix:task_cleanroom",
            },
        )

        submission_path = evaluation._collect_submission(
            instance, cast(RemoteWorkspace, workspace)
        )

        assert submission_path.read_bytes() == b"fake-archive"
        assert len(captured) == 1
        assert "base64" not in captured[0]

    def test_rejects_large_archive_when_only_base64_fallback_remains(
        self, tmp_path: Path
    ) -> None:
        from benchmarks.utils.models import EvalInstance
        from openhands.sdk.workspace.remote import RemoteWorkspace

        captured: list[str] = []

        def fake_execute_command(cmd: str, timeout: int = 0):
            captured.append(cmd)
            result = MagicMock()
            result.exit_code = 0
            result.stderr = ""
            if cmd.startswith("stat "):
                result.stdout = str(run_infer.BASE64_DOWNLOAD_MAX_BYTES + 1)
            else:
                result.stdout = ""
            return result

        evaluation = self._make_evaluation(tmp_path)
        workspace = MagicMock()
        workspace.execute_command = fake_execute_command
        workspace.file_download = None
        workspace.download_directory = None
        instance = EvalInstance(
            id="abishekvashok__cmatrix.5c082c6",
            data={
                "repository": "abishekvashok/cmatrix",
                "task_image": "programbench/cmatrix:task_cleanroom",
            },
        )

        with pytest.raises(RuntimeError, match="Refusing deprecated base64"):
            evaluation._collect_submission(instance, cast(RemoteWorkspace, workspace))

        assert len(captured) == 2
        assert captured[1].startswith("stat ")

    def test_base64_fallback_warns_for_tiny_archives(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from benchmarks.utils.models import EvalInstance
        from openhands.sdk.workspace.remote import RemoteWorkspace

        payload = b"tiny-archive"
        encoded_payload = base64.b64encode(payload).decode()
        captured: list[str] = []

        def fake_execute_command(cmd: str, timeout: int = 0):
            captured.append(cmd)
            result = MagicMock()
            result.exit_code = 0
            result.stderr = ""
            if cmd.startswith("stat "):
                result.stdout = str(len(payload))
            elif cmd.startswith("base64 "):
                result.stdout = encoded_payload
            else:
                result.stdout = ""
            return result

        evaluation = self._make_evaluation(tmp_path)
        workspace = MagicMock()
        workspace.execute_command = fake_execute_command
        workspace.file_download = None
        workspace.download_directory = None
        instance = EvalInstance(
            id="abishekvashok__cmatrix.5c082c6",
            data={
                "repository": "abishekvashok/cmatrix",
                "task_image": "programbench/cmatrix:task_cleanroom",
            },
        )

        with caplog.at_level("WARNING"):
            submission_path = evaluation._collect_submission(
                instance, cast(RemoteWorkspace, workspace)
            )

        assert submission_path.read_bytes() == payload
        assert any("deprecated base64 download fallback" in m for m in caplog.messages)
        assert len(captured) == 3
        assert captured[2].startswith("base64 ")


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
# Reference-diffs Stop hook (replaces the older gold-tests hook whose
# /opt/programbench-stashed-executable-do-not-modify lookup was never
# populated by the upstream cleanroom images).
# ---------------------------------------------------------------------------


def _make_eval_instance(
    instance_id: str = "abishekvashok__cmatrix.5c082c6",
    repository: str = "abishekvashok/cmatrix",
):
    """Build a minimal ``EvalInstance`` for hook-config tests."""
    from benchmarks.utils.models import EvalInstance

    return EvalInstance(
        id=instance_id,
        data={"repository": repository, "instance_id": instance_id},
    )


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
    layer can never be skipped. ``enforce_reference_diffs`` adds the
    reference-diffs hook on top, sequenced after the contract check.
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

    def test_appends_reference_diffs_hook_when_enforced(self) -> None:
        cfg = run_infer._build_stop_hook_config(
            _make_metadata_with_details(enforce_reference_diffs=True),
            _make_eval_instance(),
        )
        assert cfg is not None
        assert len(cfg.stop[0].hooks) == 2
        # Order matters: the cheap contract check must run first so the
        # reference-diffs hook never sees a missing ./executable.
        first, second = cfg.stop[0].hooks
        contract_body = run_infer.COMPILE_CONTRACT_HOOK_PATH.read_text()
        ref_body = run_infer.REFERENCE_DIFFS_HOOK_PATH.read_text()
        assert contract_body.strip() in first.command
        assert ref_body.strip() in second.command

    def test_reference_diffs_hook_omits_env_prelude(self) -> None:
        # The diffs hook used to need ``PB_REFERENCE_BINARY_PATH`` set
        # per-instance via an env-prelude (when we believed the reference
        # was at ``/workspace/<repo_name>``). After the retry-21
        # post-mortem we now know the reference always lives at
        # ``/workspace/executable.ref`` (Step 0 of the prompt does the
        # ``mv``), so the hook bakes that as a default and the env-prelude
        # is gone. This test pins that there's no stray
        # ``PB_REFERENCE_BINARY_PATH=...`` assignment before ``bash -s``,
        # regardless of whether an instance is supplied.
        for instance in (
            _make_eval_instance(repository="abishekvashok/cmatrix"),
            None,
        ):
            cfg = run_infer._build_stop_hook_config(
                _make_metadata_with_details(enforce_reference_diffs=True),
                instance,
            )
            assert cfg is not None
            diffs_cmd = cfg.stop[0].hooks[1].command
            prelude = diffs_cmd.split("bash -s", 1)[0]
            # The script BODY references PB_REFERENCE_BINARY_PATH in its
            # own ``${VAR:-default}`` expansion; the env-PRELUDE (before
            # ``bash -s``) must not set it.
            assert "PB_REFERENCE_BINARY_PATH=" not in prelude
            # And the hook body must default to executable.ref, matching
            # the prompt's Step 0.
            assert "PB_REFERENCE_BINARY_PATH:-/workspace/executable.ref" in diffs_cmd

    def test_does_not_install_reference_diffs_hook_by_default(self) -> None:
        cfg = run_infer._build_stop_hook_config(
            _make_metadata_with_details(), _make_eval_instance()
        )
        assert cfg is not None
        ref_body = run_infer.REFERENCE_DIFFS_HOOK_PATH.read_text()
        for hook in cfg.stop[0].hooks:
            assert ref_body.strip() not in hook.command

    def test_respects_custom_contract_timeout(self) -> None:
        cfg = run_infer._build_stop_hook_config(
            _make_metadata_with_details(compile_contract_hook_timeout=11)
        )
        assert cfg is not None
        assert cfg.stop[0].hooks[0].timeout == 11

    def test_respects_custom_reference_diffs_timeout(self) -> None:
        cfg = run_infer._build_stop_hook_config(
            _make_metadata_with_details(
                enforce_reference_diffs=True, reference_diffs_hook_timeout=42
            ),
            _make_eval_instance(),
        )
        assert cfg is not None
        assert cfg.stop[0].hooks[1].timeout == 42

    def test_only_stop_event_is_populated(self) -> None:
        # Sanity: we don't accidentally wire the scripts to fire on
        # every tool invocation.
        cfg = run_infer._build_stop_hook_config(
            _make_metadata_with_details(enforce_reference_diffs=True),
            _make_eval_instance(),
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
    """Set up a fake workspace + state dir for the bash hooks.

    ``runs_dir`` is deliberately a sibling of ``workspace`` (not a
    subdirectory) so the hook's state mirrors the production default
    (``/tmp/...``) rather than the old in-workspace location. The
    workspace-isolation contract — see ``test_does_not_mutate_workspace``
    in ``TestCompileContractHookScript`` — requires that the hook
    leaves ``$WORKSPACE`` byte-stable."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    eval_dir = workspace / "eval"
    eval_dir.mkdir()
    runs_dir = tmp_path / "stop-hook-state"
    return {
        "workspace": workspace,
        "eval": eval_dir,
        "runs_dir": runs_dir,
        "agent": workspace / "executable",
        "reference": tmp_path / "reference_bin",
    }


def _run_hook(
    script: Path,
    cwd: Path,
    env_overrides: dict[str, str],
    stdin: str = "{}",
):
    """Invoke the hook script with isolated env vars.

    ``env_overrides`` is passed verbatim to the subprocess; only ``PATH``
    is preserved by default so coreutils/timeout/diff resolve. Drop
    everything else so a developer's locally-set ``PB_*`` doesn't leak
    in and skew the assertions.
    """
    import os
    import subprocess

    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
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


def _build_help_binary(
    src_path: Path,
    out_path: Path,
    *,
    help_text: str,
    short_help_text: str | None = None,
) -> None:
    """Compile a tiny C ``--help`` / ``-h`` echo binary.

    The hook script's whole job is byte-comparing those flags' output;
    using a real (compiled, executable) binary instead of a shell wrapper
    keeps the test exercising the same ``execve``-based code path the
    production hook hits.
    """
    import shutil
    import subprocess

    if shutil.which("gcc") is None:
        pytest.skip(
            "gcc not available; reference-diffs hook tests need to compile a tiny C binary"
        )
    short = short_help_text if short_help_text is not None else help_text
    src_path.write_text(
        "#include <stdio.h>\n"
        "#include <string.h>\n"
        "int main(int argc, char **argv) {\n"
        '    if (argc > 1 && strcmp(argv[1], "--help") == 0) {\n'
        f"        fputs({json.dumps(help_text)}, stdout);\n"
        "        return 0;\n"
        "    }\n"
        '    if (argc > 1 && strcmp(argv[1], "-h") == 0) {\n'
        f"        fputs({json.dumps(short)}, stdout);\n"
        "        return 0;\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    )
    subprocess.run(
        ["gcc", "-O0", "-o", str(out_path), str(src_path)],
        check=True,
        capture_output=True,
    )
    out_path.chmod(0o755)


class TestReferenceDiffsHookScript:
    """End-to-end tests of the new reference-diffs Stop hook.

    The hook compares the agent's binary's ``--help`` / ``-h`` output
    against the reference binary's, byte-for-byte. We give it real
    compiled C binaries (one matching, one diverging by a single
    character) and check it blocks/allows correctly.
    """

    SCRIPT = run_infer.REFERENCE_DIFFS_HOOK_PATH

    def _common_env(self, hook_sandbox) -> dict[str, str]:
        return {
            "PB_REFERENCE_BINARY_PATH": str(hook_sandbox["reference"]),
            "PB_AGENT_BINARY_PATH": str(hook_sandbox["agent"]),
            "PB_REFERENCE_DIFFS_RUNS_DIR": str(hook_sandbox["runs_dir"]),
            "PB_REFERENCE_DIFFS_MAX_RETRIES": "3",
            "PB_REFERENCE_DIFFS_TIMEOUT": "10",
        }

    def test_allows_stop_when_help_matches_byte_for_byte(self, hook_sandbox, tmp_path):
        same = " Usage: ref [-abc]\n"
        _build_help_binary(
            tmp_path / "ref.c", hook_sandbox["reference"], help_text=same
        )
        _build_help_binary(tmp_path / "agent.c", hook_sandbox["agent"], help_text=same)
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides=self._common_env(hook_sandbox),
        )
        assert result.returncode == 0, result.stderr
        assert "match reference" in result.stderr

    def test_blocks_stop_on_leading_space_drift(self, hook_sandbox, tmp_path):
        # The exact failure mode that hit cmatrix: reference's help banner
        # starts with " Usage:" (leading space) but the agent's prints
        # "Usage:". Single-character drift, score-killing.
        ref_help = " Usage: ref [-abc]\n"
        agent_help = "Usage: ref [-abc]\n"
        _build_help_binary(
            tmp_path / "ref.c", hook_sandbox["reference"], help_text=ref_help
        )
        _build_help_binary(
            tmp_path / "agent.c", hook_sandbox["agent"], help_text=agent_help
        )
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides=self._common_env(hook_sandbox),
        )
        assert result.returncode == 2, result.stderr
        assert "differs from the reference" in result.stderr
        # The unified diff is what gives the agent something actionable.
        assert "--help" in result.stderr
        assert "Usage: ref" in result.stderr

    def test_blocks_stop_on_invented_banner_line(self, hook_sandbox, tmp_path):
        # The exact failure mode that hit zip-password-finder: the agent
        # added a "Targeting file '...'" preamble that the reference
        # never prints. Even when the rest matches.
        ref_help = "ref 1.0\n"
        agent_help = "Starting up...\nref 1.0\n"
        _build_help_binary(
            tmp_path / "ref.c", hook_sandbox["reference"], help_text=ref_help
        )
        _build_help_binary(
            tmp_path / "agent.c", hook_sandbox["agent"], help_text=agent_help
        )
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides=self._common_env(hook_sandbox),
        )
        assert result.returncode == 2, result.stderr
        assert "Starting up" in result.stderr

    def test_allows_stop_when_reference_binary_missing(self, hook_sandbox, tmp_path):
        # If the reference somehow isn't present in the cleanroom image,
        # we have nothing to compare against and must fall back to
        # allow-stop (the upstream eval is the source of truth).
        _build_help_binary(tmp_path / "agent.c", hook_sandbox["agent"], help_text="x\n")
        env = self._common_env(hook_sandbox)
        env["PB_REFERENCE_BINARY_PATH"] = "/nonexistent/reference"
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides=env,
        )
        assert result.returncode == 0, result.stderr
        assert "not an executable file" in result.stderr

    def test_default_reference_path_is_executable_ref(self, hook_sandbox, tmp_path):
        # When PB_REFERENCE_BINARY_PATH isn't set in the env, the hook
        # defaults to /workspace/executable.ref -- matching the path
        # Step 0 of the prompt tells the agent to ``mv`` the cleanroom
        # reference into. In the test sandbox that path won't exist, so
        # we still get an allow-stop fallback, but the stderr message
        # has to mention executable.ref so we know we hit the default.
        _build_help_binary(tmp_path / "agent.c", hook_sandbox["agent"], help_text="x\n")
        env = self._common_env(hook_sandbox)
        env.pop("PB_REFERENCE_BINARY_PATH", None)
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides=env,
        )
        assert result.returncode == 0, result.stderr
        # Default should be honoured, not silently swapped:
        assert "/workspace/executable.ref" in result.stderr
        assert "is not an executable file" in result.stderr

    def test_allows_stop_when_agent_binary_missing(self, hook_sandbox, tmp_path):
        # Compile-contract hook owns the missing-./executable error path;
        # the diffs hook must defer rather than double-block (which would
        # eat into the retry cap before the agent sees the contract msg).
        _build_help_binary(
            tmp_path / "ref.c", hook_sandbox["reference"], help_text="x\n"
        )
        # Note: hook_sandbox["agent"] does NOT exist on disk.
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides=self._common_env(hook_sandbox),
        )
        assert result.returncode == 0, result.stderr
        assert "deferring to compile-contract hook" in result.stderr

    def test_retry_cap_lets_agent_eventually_stop(self, hook_sandbox, tmp_path):
        # If the agent stays broken, the hook must eventually concede so
        # we don't burn the entire iteration budget on stop hooks.
        _build_help_binary(
            tmp_path / "ref.c", hook_sandbox["reference"], help_text="ref\n"
        )
        _build_help_binary(
            tmp_path / "agent.c", hook_sandbox["agent"], help_text="other\n"
        )
        env = self._common_env(hook_sandbox)
        env["PB_REFERENCE_DIFFS_MAX_RETRIES"] = "2"
        # First two invocations block...
        for attempt in (1, 2):
            r = _run_hook(
                self.SCRIPT,
                cwd=hook_sandbox["workspace"],
                env_overrides=dict(env),
            )
            assert r.returncode == 2, (
                f"attempt {attempt} should block; stderr={r.stderr}"
            )
        # Third invocation hits the cap and concedes.
        r = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides=dict(env),
        )
        assert r.returncode == 0, r.stderr
        assert "max retries" in r.stderr

    def test_does_not_mutate_workspace(self, hook_sandbox, tmp_path):
        # The orchestrator tars /workspace immediately after the hook
        # returns; any churn there would race with tar. Assert the
        # workspace's mtime + listing is byte-stable across the hook.
        _build_help_binary(
            tmp_path / "ref.c", hook_sandbox["reference"], help_text="x\n"
        )
        _build_help_binary(tmp_path / "agent.c", hook_sandbox["agent"], help_text="x\n")
        before = sorted(
            (p.relative_to(hook_sandbox["workspace"]), p.stat().st_size)
            for p in hook_sandbox["workspace"].rglob("*")
            if p.is_file()
        )
        _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides=self._common_env(hook_sandbox),
        )
        after = sorted(
            (p.relative_to(hook_sandbox["workspace"]), p.stat().st_size)
            for p in hook_sandbox["workspace"].rglob("*")
            if p.is_file()
        )
        assert before == after, (
            "reference-diffs hook mutated the workspace; this races with "
            "the orchestrator's submission tar invocation"
        )

    def test_truncates_runaway_diff(self, hook_sandbox, tmp_path):
        # If the agent emits megabytes of garbage on --help (e.g. a
        # debug dump), the hook must NOT pipe all of it back to the
        # agent — that would bury the conversation in one event.
        ref_help = "x\n"
        agent_help = "y" * 20000 + "\n"
        _build_help_binary(
            tmp_path / "ref.c", hook_sandbox["reference"], help_text=ref_help
        )
        _build_help_binary(
            tmp_path / "agent.c", hook_sandbox["agent"], help_text=agent_help
        )
        env = self._common_env(hook_sandbox)
        env["PB_REFERENCE_DIFFS_MAX_DIFF_BYTES"] = "500"
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides=env,
        )
        assert result.returncode == 2, result.stderr
        assert "diff truncated" in result.stderr


# ---------------------------------------------------------------------------
# Reference-diffs hook v2: subcommand discovery + argv[0] normalization
# ---------------------------------------------------------------------------


def _build_subcommand_binary(
    src_path: Path,
    out_path: Path,
    *,
    top_help: str,
    subcommands: dict[str, dict[str, object]],
    use_argv0_in_help: bool = False,
) -> None:
    """Compile a tiny C binary that mimics a clap-style multi-subcommand CLI.

    ``subcommands`` is keyed by subcommand name and each value is a dict
    with optional keys:
      * ``help``: stdout text for ``<sub> --help`` (rc 0)
      * ``error_text``: stderr text for any other ``<sub> ...`` invocation
      * ``error_rc``: rc for the same (default 0)

    The point is to exercise three v2 probe paths:
      1. subcommand --help byte-comparison
      2. subcommand <bogus-flag> -> rc/stderr divergence
      3. subcommand <bogus-path> -> rc/stderr divergence

    ``use_argv0_in_help=True`` makes top-level help include the program's
    argv[0] basename, used to verify argv[0] normalization works for
    real ELF binaries.
    """
    import shutil
    import subprocess

    if shutil.which("gcc") is None:
        pytest.skip(
            "gcc not available; reference-diffs v2 hook tests need to compile a tiny C binary"
        )

    # Build the C source. The big switch over subcommand names matches by
    # strcmp; we compose it with json.dumps for safe quoting.
    sub_blocks = []
    for name, spec in subcommands.items():
        # ``spec`` is dict[str, object] for flexibility; coerce here so
        # we get pyright-clean str/int locals that can flow into the
        # f-string below.
        help_text = str(spec.get("help", f"Usage: ... {name}"))
        err_text = str(spec.get("error_text", ""))
        err_rc_raw = spec.get("error_rc", 0)
        err_rc = int(err_rc_raw) if isinstance(err_rc_raw, (int, str)) else 0
        sub_blocks.append(
            f"    if (strcmp(argv[1], {json.dumps(name)}) == 0) {{\n"
            f'        if (argc > 2 && strcmp(argv[2], "--help") == 0) {{\n'
            f"            fputs({json.dumps(help_text)}, stdout);\n"
            f"            return 0;\n"
            f"        }}\n"
            f"        fputs({json.dumps(err_text)}, stderr);\n"
            f"        return {err_rc};\n"
            f"    }}\n"
        )

    if use_argv0_in_help:
        # bn = basename(argv[0]), then printf("...%s...", bn)
        help_section = (
            "        char buf[256];\n"
            "        strncpy(buf, argv[0], sizeof(buf)-1); buf[sizeof(buf)-1] = 0;\n"
            "        char *bn = basename(buf);\n"
            f"        printf({json.dumps(top_help)}, bn);\n"
        )
        extra_includes = "#include <libgen.h>\n"
    else:
        help_section = f"        fputs({json.dumps(top_help)}, stdout);\n"
        extra_includes = ""

    src_path.write_text(
        "#include <stdio.h>\n"
        "#include <string.h>\n"
        "#include <stdlib.h>\n" + extra_includes + "int main(int argc, char **argv) {\n"
        '    if (argc > 1 && (strcmp(argv[1], "--help") == 0\n'
        '                  || strcmp(argv[1], "-h") == 0)) {\n'
        + help_section
        + "        return 0;\n"
        "    }\n" + "".join(sub_blocks) + '    fputs("error: unknown\\n", stderr);\n'
        "    return 2;\n"
        "}\n"
    )
    subprocess.run(
        ["gcc", "-O0", "-o", str(out_path), str(src_path)],
        check=True,
        capture_output=True,
    )
    out_path.chmod(0o755)


class TestReferenceDiffsHookV2:
    """v2 probe suite: subcommand discovery + invalid-input + argv[0] norm.

    Post-retry-22 we extended the hook to discover subcommands from the
    reference's top-level ``--help`` output and probe each one with
    ``--help``, an invalid flag, and a nonexistent path. We also wrap
    invocation in ``exec -a`` so both binaries see the same argv[0] —
    this prevents false positives when both ref and agent correctly
    derive their ``Usage:`` line from argv[0] but happen to live at
    different paths on disk (the realistic ProgramBench layout).
    """

    SCRIPT = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "programbench"
        / "hooks"
        / "check_reference_diffs.sh"
    )

    def _common_env(self, hook_sandbox) -> dict[str, str]:
        return {
            "PB_REFERENCE_BINARY_PATH": str(hook_sandbox["reference"]),
            "PB_AGENT_BINARY_PATH": str(hook_sandbox["agent"]),
            "PB_REFERENCE_DIFFS_RUNS_DIR": str(hook_sandbox["runs_dir"]),
            "PB_WORKSPACE": str(hook_sandbox["workspace"]),
        }

    # Reference top-level help text: a clap-shaped Commands: section
    # listing two subcommands plus the auto-help one (which the hook's
    # parser must filter out). We pin the exact format because the awk
    # parser is heuristic and a future change here could regress
    # discovery silently.
    REF_TOP_HELP = (
        "A test reference binary\n"
        "\n"
        "Usage: executable [COMMAND]\n"
        "\n"
        "Commands:\n"
        "  add     Add an entry\n"
        "  remove  Remove an entry\n"
        "  help    Print help\n"
        "\n"
        "Options:\n"
        "  -h, --help  Print help\n"
    )

    def test_blocks_on_subcommand_help_drift(self, hook_sandbox, tmp_path):
        # Top-level help matches; a SUBCOMMAND'S --help drifts. v1
        # would have allowed stop (it only diffed top-level). v2 must
        # block.
        _build_subcommand_binary(
            tmp_path / "ref.c",
            hook_sandbox["reference"],
            top_help=self.REF_TOP_HELP,
            subcommands={
                "add": {"help": "Usage: executable add <DIR>\n"},
                "remove": {"help": "Usage: executable remove <DIR>\n"},
            },
        )
        _build_subcommand_binary(
            tmp_path / "agent.c",
            hook_sandbox["agent"],
            top_help=self.REF_TOP_HELP,  # top-level matches
            subcommands={
                "add": {
                    # --- drift: agent emits an extra "Args:" line that
                    # the reference doesn't have. ---
                    "help": "Usage: executable add <DIR>\nArgs:\n  <DIR>  the directory\n",
                },
                "remove": {"help": "Usage: executable remove <DIR>\n"},
            },
        )
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides=self._common_env(hook_sandbox),
        )
        assert result.returncode == 2, result.stderr
        assert "subcommand `add --help`" in result.stderr
        # The diff body must include the line that drifted, so the agent
        # can act on it without re-running the probe themselves.
        assert "Args:" in result.stderr

    def test_blocks_on_missing_subcommand_validation(self, hook_sandbox, tmp_path):
        # Reference rc=1 with a "not a directory" stderr on bad input;
        # agent rc=0 with empty output (the silent-success bug). v1
        # would have missed this entirely (top-level help is identical);
        # v2's invalid-flag and nonexistent-path probes catch it.
        _build_subcommand_binary(
            tmp_path / "ref.c",
            hook_sandbox["reference"],
            top_help=self.REF_TOP_HELP,
            subcommands={
                "add": {
                    "help": "Usage: executable add <DIR>\n",
                    "error_text": "error: not a directory\n",
                    "error_rc": 1,
                },
                "remove": {
                    "help": "Usage: executable remove <DIR>\n",
                    "error_text": "error: entry not found\n",
                    "error_rc": 1,
                },
            },
        )
        _build_subcommand_binary(
            tmp_path / "agent.c",
            hook_sandbox["agent"],
            top_help=self.REF_TOP_HELP,
            subcommands={
                "add": {
                    "help": "Usage: executable add <DIR>\n",
                    "error_text": "",
                    "error_rc": 0,  # <-- BUG: silent success
                },
                "remove": {
                    "help": "Usage: executable remove <DIR>\n",
                    "error_text": "",
                    "error_rc": 0,  # <-- BUG: silent success
                },
            },
        )
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides=self._common_env(hook_sandbox),
        )
        assert result.returncode == 2, result.stderr
        # Both invalid-flag and nonexistent-path probes catch the rc divergence.
        assert "invalid flag" in result.stderr
        assert "nonexistent path" in result.stderr
        # And the rc=1 vs rc=0 must be visible in the diff header.
        assert "rc: ref=1, agent=0" in result.stderr

    def test_argv0_normalization_avoids_basename_false_positive(
        self, hook_sandbox, tmp_path
    ):
        # Both binaries derive their ``Usage:`` line from argv[0] (the
        # clap-default behaviour). They are byte-identical implementations
        # but live at different paths on disk. v1 would diff their
        # outputs and find ``Usage: ref [COMMAND]`` vs ``Usage: agent
        # [COMMAND]`` — a FALSE POSITIVE. v2 wraps both in ``exec -a
        # executable`` so argv[0] is normalised and the outputs match.
        # NOTE: top_help must contain a single ``%s`` for the argv[0]
        # basename when ``use_argv0_in_help=True``.
        argv0_help_template = (
            "A test\n"
            "\n"
            "Usage: %s [COMMAND]\n"
            "\n"
            "Commands:\n"
            "  add  Add an entry\n"
            "  help Print help\n"
        )
        _build_subcommand_binary(
            tmp_path / "ref.c",
            hook_sandbox["reference"],
            top_help=argv0_help_template,
            subcommands={"add": {"help": "Usage: add <DIR>\n"}},
            use_argv0_in_help=True,
        )
        _build_subcommand_binary(
            tmp_path / "agent.c",
            hook_sandbox["agent"],
            top_help=argv0_help_template,
            subcommands={"add": {"help": "Usage: add <DIR>\n"}},
            use_argv0_in_help=True,
        )
        # Sanity: rename so the two binaries have OBVIOUSLY different
        # paths-on-disk. If exec -a normalisation is broken, the hook
        # WOULD see ``Usage: ref_bin`` vs ``Usage: agent_bin`` and block.
        ref_renamed = hook_sandbox["reference"].parent / "ref_bin"
        agent_renamed = hook_sandbox["agent"].parent / "agent_bin"
        hook_sandbox["reference"].rename(ref_renamed)
        hook_sandbox["agent"].rename(agent_renamed)
        env = self._common_env(hook_sandbox)
        env["PB_REFERENCE_BINARY_PATH"] = str(ref_renamed)
        env["PB_AGENT_BINARY_PATH"] = str(agent_renamed)
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides=env,
        )
        assert result.returncode == 0, (
            f"argv[0] normalisation regressed; rc={result.returncode}\n"
            f"stderr=\n{result.stderr}"
        )
        # Sanity: it still verified the probes (didn't no-op).
        assert "all" in result.stderr and "comparable probe" in result.stderr

    def test_no_commands_section_falls_back_to_toplevel_only(
        self, hook_sandbox, tmp_path
    ):
        # Reference has NO ``Commands:`` section in its --help output
        # (single-purpose binary like cmatrix). The discovery awk must
        # emit zero subcommands and the hook must still allow stop on
        # matching top-level help.
        plain_help = (
            "A single-purpose tool\n"
            "\n"
            "Usage: executable [-abc] [-C COLOR]\n"
            "\n"
            "Options:\n"
            "  -a   thing a\n"
            "  -C   colour\n"
        )
        _build_subcommand_binary(
            tmp_path / "ref.c",
            hook_sandbox["reference"],
            top_help=plain_help,
            subcommands={},
        )
        _build_subcommand_binary(
            tmp_path / "agent.c",
            hook_sandbox["agent"],
            top_help=plain_help,
            subcommands={},
        )
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides=self._common_env(hook_sandbox),
        )
        assert result.returncode == 0, result.stderr
        # Three top-level probes (--help, -h, top-level invalid flag),
        # zero subcommand probes. Pin that we did NOT silently skip them.
        assert "all 3 comparable probe" in result.stderr or (
            "all" in result.stderr and "comparable probe" in result.stderr
        )

    def test_caps_subcommand_count_to_avoid_runaway(self, hook_sandbox, tmp_path):
        # If the binary has 30 subcommands, we must cap the probe count.
        # Each subcommand gets up to 3 probes (--help, invalid flag,
        # nonexistent path) but the bogus-flag/path probes are SKIPPED
        # by the hook when the reference's stderr is empty (nothing
        # meaningful to compare). We give every subcommand non-empty
        # error text so all 3 probes contribute, and pin the cap envvar
        # so this test doesn't depend on the default drifting.
        many_help = "Tool\n\nUsage: executable [CMD]\n\nCommands:\n"
        names = [f"cmd{i:02d}" for i in range(30)]
        for n in names:
            many_help += f"  {n}  Do {n}\n"
        many_help += "\nOptions:\n  -h  help\n"
        subs = {
            n: {
                "help": f"Usage: executable {n}\n",
                "error_text": f"error: bad {n} args\n",
                "error_rc": 1,
            }
            for n in names
        }
        _build_subcommand_binary(
            tmp_path / "ref.c",
            hook_sandbox["reference"],
            top_help=many_help,
            subcommands=subs,
        )
        # Agent: identical → no diffs expected, just verify count.
        _build_subcommand_binary(
            tmp_path / "agent.c",
            hook_sandbox["agent"],
            top_help=many_help,
            subcommands=subs,
        )
        env = self._common_env(hook_sandbox)
        env["PB_REFERENCE_DIFFS_MAX_SUBCMDS"] = "5"
        result = _run_hook(
            self.SCRIPT,
            cwd=hook_sandbox["workspace"],
            env_overrides=env,
        )
        assert result.returncode == 0, result.stderr
        # 3 top-level + 5 subcommands * 3 probes = 18 total. If the cap
        # weren't honoured this would be 3 + 30*3 = 93.
        assert "all 18 comparable probe" in result.stderr, result.stderr


# ---------------------------------------------------------------------------
# SDK Stop-hook contract pin
# ---------------------------------------------------------------------------


class TestStopHookSdkContract:
    """Both Stop hooks MUST exit 2 (not 1) on the block path.

    The OpenHands Software Agent SDK's hook executor only treats
    ``exit_code == 2`` as a Stop block (see
    ``software-agent-sdk/openhands-sdk/openhands/sdk/hooks/executor.py``
    where ``blocked = (returncode == 2)``).  Any other non-zero exit is
    logged as ``Stop hook error`` and the agent is allowed to stop.

    retry-14 hit exactly this: every block path used ``exit 1`` and the
    SDK silently let the agent ship a broken submission, producing
    ``Resolved 0 / 3 (almost: 1, errors: 0)`` with no signs of agent
    iteration.  These assertions exist so a future refactor can't
    quietly regress to ``exit 1``.
    """

    REF_DIFFS = Path("benchmarks/programbench/hooks/check_reference_diffs.sh")
    COMPILE = Path("benchmarks/programbench/hooks/check_compile_contract.sh")

    @pytest.mark.parametrize("script", [REF_DIFFS, COMPILE])
    def test_block_paths_exit_two_not_one(self, script: Path) -> None:
        text = script.read_text()
        # Any standalone ``exit 1`` is a smell -- the SDK ignores it.
        offending = [
            (i + 1, line)
            for i, line in enumerate(text.splitlines())
            if re.match(r"^\s*exit 1\s*$", line)
        ]
        assert offending == [], (
            f"{script}: {len(offending)} `exit 1` line(s) -- the SDK only "
            "blocks on rc=2, so these silently let the agent stop. "
            f"Offending lines: {offending}"
        )

    @pytest.mark.parametrize("script", [REF_DIFFS, COMPILE])
    def test_documents_exit_two_contract(self, script: Path) -> None:
        # Force future maintainers to read about the contract before
        # editing the hook.
        text = script.read_text()
        assert "exit 2" in text, (
            f"{script}: should declare ``exit 2 -> block`` in its header "
            "comment so the SDK contract is discoverable."
        )


class TestHooksRunUnderSdkHeredocWrap:
    """Both Stop hooks MUST actually execute when wrapped the way the SDK
    runs them in production.

    ``run_infer.py::_hook_definition_from_script`` packages each hook as

        bash -s <<'PROGRAMBENCH_HOOK_EOF'
        <hook script body>
        PROGRAMBENCH_HOOK_EOF

    and the SDK's ``HookExecutor.execute`` invokes that string via
    ``subprocess.run(..., shell=True, input=event_json)``. Under
    ``bash -s`` bash reads the script body itself from stdin (the
    heredoc), which means the hook MUST NOT consume or redirect its
    own stdin -- doing so swallows the rest of the script source and
    bash silently exits 0 before any of the contract checks run,
    turning the hook into a no-op that green-lights every broken
    submission.

    retry-15 hit exactly this: a ``cat >/dev/null`` line at the top
    of each hook (intended to "drain stdin so the SDK doesn't see a
    SIGPIPE") consumed the rest of the heredoc, the hooks reported
    ``exit_code=0, stdout='', stderr=''`` for every Stop event, and
    the agent shipped 0/3 with the (then-named) gold-tests hook
    supposedly enabled. The existing
    ``TestReferenceDiffsHookScript`` / ``TestCompileContractHookScript``
    suites couldn't see this because they invoke the script as
    ``bash <file>`` rather than via the SDK's heredoc wrap.

    These tests close that gap by exercising the *actual* wrap.
    """

    REF_DIFFS = run_infer.REFERENCE_DIFFS_HOOK_PATH
    COMPILE = run_infer.COMPILE_CONTRACT_HOOK_PATH

    @staticmethod
    def _wrap(script_path: Path) -> str:
        """Mirror ``run_infer.py::_hook_definition_from_script`` exactly."""
        body = script_path.read_text()
        return f"bash -s <<'PROGRAMBENCH_HOOK_EOF'\n{body}\nPROGRAMBENCH_HOOK_EOF\n"

    @staticmethod
    def _run_wrapped(
        script_path: Path,
        *,
        env_overrides: dict[str, str],
        cwd: Path,
        stdin: str = '{"reason":"agent_finished","event_type":"Stop"}',
        timeout: int = 30,
    ):
        """Run the heredoc-wrapped hook the way the SDK does."""
        import os
        import subprocess

        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        }
        env.update({k: str(v) for k, v in env_overrides.items()})
        return subprocess.run(
            TestHooksRunUnderSdkHeredocWrap._wrap(script_path),
            shell=True,
            cwd=str(cwd),
            env=env,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def test_compile_hook_blocks_missing_compile_sh_via_heredoc_wrap(
        self, tmp_path: Path
    ) -> None:
        """Smoking gun for retry-15: empty workspace + no compile.sh
        MUST produce rc=2 with explanatory stderr -- not silent rc=0
        with empty output.
        """
        workspace = tmp_path / "ws"
        workspace.mkdir()
        runs = tmp_path / "runs"
        result = self._run_wrapped(
            self.COMPILE,
            cwd=workspace,
            env_overrides={
                "PB_WORKSPACE": str(workspace),
                "PB_COMPILE_HOOK_RUNS_DIR": str(runs),
                "PB_COMPILE_HOOK_MAX_RETRIES": "3",
                "PB_COMPILE_HOOK_TIMEOUT": "30",
            },
        )
        # rc=2 because the workspace is missing compile.sh.
        assert result.returncode == 2, (
            "compile-contract hook silently exited rc="
            f"{result.returncode} under SDK heredoc wrap; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        # And it must produce real feedback the SDK can route back to
        # the agent. Empty stderr was the smoking gun in retry-15.
        assert result.stderr.strip(), (
            "compile-contract hook produced no stderr under SDK wrap; "
            "this is the retry-15 self-termination footprint."
        )
        assert "compile.sh is missing" in result.stderr

    def test_reference_diffs_hook_blocks_on_help_drift_via_heredoc_wrap(
        self, tmp_path: Path
    ) -> None:
        """Same regression, reference-diffs side: a single-byte help
        drift between the agent and reference MUST block with rc=2
        even under the heredoc wrap.
        """
        workspace = tmp_path / "ws"
        workspace.mkdir()
        runs = tmp_path / "runs"
        reference = tmp_path / "ref"
        agent = workspace / "executable"
        # Use the same compiled-binary fixture the rest of the suite uses.
        _build_help_binary(tmp_path / "ref.c", reference, help_text=" ref\n")
        _build_help_binary(tmp_path / "agent.c", agent, help_text="ref\n")
        result = self._run_wrapped(
            self.REF_DIFFS,
            cwd=workspace,
            env_overrides={
                "PB_REFERENCE_BINARY_PATH": str(reference),
                "PB_AGENT_BINARY_PATH": str(agent),
                "PB_REFERENCE_DIFFS_RUNS_DIR": str(runs),
                "PB_REFERENCE_DIFFS_MAX_RETRIES": "3",
                "PB_REFERENCE_DIFFS_TIMEOUT": "10",
            },
        )
        assert result.returncode == 2, (
            "reference-diffs hook silently exited rc="
            f"{result.returncode} under SDK heredoc wrap; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert result.stderr.strip(), (
            "reference-diffs hook produced no stderr under SDK wrap; "
            "this is the retry-15 self-termination footprint."
        )
        assert "differs from the reference" in result.stderr

    @pytest.mark.parametrize("script", [REF_DIFFS, COMPILE])
    def test_hooks_do_not_consume_their_own_stdin(self, script: Path) -> None:
        """Static guard: forbid stdin-consuming patterns at the top
        level of any hook script.

        Under ``bash -s`` + heredoc, anything that consumes stdin
        consumes the script source itself. The forbidden patterns
        below all share that footprint:

          * ``cat`` with no file argument and no input redirect
          * ``exec </dev/null`` (or any ``exec <…``)
          * unguarded ``read line``

        We only inspect non-comment lines: comments are *about* the
        contract, not violations of it.
        """
        body = script.read_text()
        offenders: list[tuple[int, str]] = []
        for lineno, raw in enumerate(body.splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # cat with no file argument (would read from stdin)
            if re.match(r"^cat(\s+[<>][^\s]+)*\s*(\|.*)?$", line):
                offenders.append((lineno, raw))
            elif re.match(r"^cat\s+([<>][^\s]+\s*)+$", line):
                offenders.append((lineno, raw))
            # exec </anything — redirects bash's own stdin
            elif re.match(r"^exec\s+<", line):
                offenders.append((lineno, raw))
            # read at top-level (subshell/while loops have their own
            # stdin scope; we only catch bare ``read VAR`` /
            # ``read -r VAR``).  ``\bread\b`` ensures we don't fire on
            # ``readonly`` / ``readline``.
            elif re.match(r"^read(\s+|$)", line):
                offenders.append((lineno, raw))
        assert offenders == [], (
            f"{script}: top-level statements that read or redirect "
            "stdin -- under ``bash -s`` + heredoc those swallow the "
            "rest of THIS script's source and bash silently exits 0 "
            "before any contract check runs.  Offenders: "
            f"{offenders}"
        )


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
    """Invoke ``check_compile_contract.sh`` against an isolated workspace.

    The default ``runs_dir`` is a sibling of ``workspace`` rather than a
    subdirectory, mirroring the production default (``/tmp/...``). This
    matters because the workspace-isolation contract requires runs_dir
    to live outside ``$WORKSPACE``."""
    import os
    import subprocess

    runs_dir = runs_dir or (workspace.parent / ".programbench-compile-hook-state")
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
        assert r.returncode == 2, r.stderr
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
        assert r.returncode == 2, r.stderr
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
        assert r.returncode == 2, r.stderr
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
        # The verification log lists the scratch dir, confirming the
        # hook actually executed compile.sh (rather than short-circuiting).
        assert "verified in /tmp/" in r.stderr

    def test_does_not_mutate_workspace(self, tmp_path: Path) -> None:
        """Workspace-isolation contract: regardless of what compile.sh
        does, the hook must leave $WORKSPACE byte-stable so the
        orchestrator's submission tarball can't race with our build
        artifacts (``tar: .: file changed as we read it``).

        Pins the fix that retired in-workspace compilation. Earlier
        versions of this hook ran compile.sh directly in $WORKSPACE,
        which left target/, build/, and ./executable behind and tripped
        ``tar: .: file changed as we read it`` mid-snapshot."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        # A noisy compile.sh that creates several files and dirs in
        # the cwd. None of these should bleed into $WORKSPACE.
        (workspace / "compile.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "mkdir -p target/release build\n"
            "echo objfile > build/foo.o\n"
            "echo binary  > target/release/foo\n"
            'printf "#!/usr/bin/env bash\\nexit 0\\n" > ./executable\n'
            "chmod +x ./executable\n"
        )
        (workspace / "compile.sh").chmod(0o755)
        before = sorted(p.name for p in workspace.iterdir())

        r = _run_compile_hook(workspace)
        assert r.returncode == 0, r.stderr

        after = sorted(p.name for p in workspace.iterdir())
        assert before == after, (
            "compile-contract hook leaked artifacts into $WORKSPACE "
            f"(before={before}, after={after}). This will trip "
            "'tar: .: file changed as we read it' when the orchestrator "
            "snapshots the submission."
        )

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
        assert r.returncode == 2, r.stderr
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
        # First three calls block (rc=2, contract not satisfied); the
        # fourth trips the retry cap and releases the agent.
        assert [r.returncode for r in runs[:3]] == [2, 2, 2]
        assert runs[3].returncode == 0, runs[3].stderr
        assert "max retries" in runs[3].stderr


# ---------------------------------------------------------------------------
# CLI flag plumbing
# ---------------------------------------------------------------------------


class TestCondenserCliPlumbing:
    """`--condenser-max-size` etc. used to be parsed but ignored. These
    tests pin the shape we now expect: the parser accepts them, defaults
    are well-defined, and `--enforce-reference-diffs` shows up alongside
    the pre-existing flags."""

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
        # Mirror run_infer.main(): enforce-reference-diffs is on by
        # default (helm dispatch can't pass extra CLI args, so the
        # default is the only switch we have for production runs).
        parser.add_argument(
            "--enforce-reference-diffs",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        parser.add_argument("--reference-diffs-hook-timeout", type=int, default=120)
        parser.set_defaults(**INFER_DEFAULTS)

        # Assertion 1: argparse can parse a leaderboard-style invocation
        # without flipping the reference-diffs default, and the args
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
                "--reference-diffs-hook-timeout",
                "180",
            ]
        )
        assert args.max_iterations == 1000
        assert args.enable_condenser is True
        assert args.condenser_max_size == 80
        assert args.condenser_keep_first == 4
        # Default-on so production helm dispatch picks up the
        # reference-diffs hook without orchestrator changes.
        assert args.enforce_reference_diffs is True
        assert args.reference_diffs_hook_timeout == 180

        # Assertion 2: ``--no-enforce-reference-diffs`` lets local
        # smoke runs opt out of the diff check.
        opted_out = parser.parse_args(["/dev/null", "--no-enforce-reference-diffs"])
        assert opted_out.enforce_reference_diffs is False
        # Sanity: didn't introduce a stray attribute that nobody owns.
        assert isinstance(parser, argparse.ArgumentParser)
        # silences "imported but unused" without changing public surface
        assert callable(main)


# ---------------------------------------------------------------------------
# _run_programbench_eval timeout safety net
# ---------------------------------------------------------------------------


class TestRunProgrambenchEvalTimeout:
    """Bound the wall clock of the ``programbench eval`` subprocess.

    We discovered on retry-16 that a hung docker container in the eval
    phase keeps the eval pod alive indefinitely (no global timeout in
    the upstream CLI, no ``activeDeadlineSeconds`` on the k8s job, no
    ``EVAL_TIMEOUT`` plumbed for programbench). The fix is a defensive
    ``subprocess.run(timeout=...)`` with a ``--eval-timeout`` CLI flag
    plumbed through the eval-job script. These tests pin both the
    happy-path return code and the timeout-kill behaviour using a
    sleep-based fake CLI on PATH.
    """

    @pytest.fixture
    def fake_cli(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Drop a fake ``programbench`` shim on PATH whose ``eval`` subcommand
        sleeps for a configurable duration. Lets us hit the timeout path
        without requiring docker or the upstream wheel."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        cli = bin_dir / "programbench"
        cli.write_text(
            "#!/usr/bin/env bash\n"
            "# Honor the test-controlled sleep duration; default 0 to keep\n"
            "# happy-path tests fast.\n"
            'sleep "${PROGRAMBENCH_FAKE_SLEEP:-0}"\n'
            'exit "${PROGRAMBENCH_FAKE_EXIT:-0}"\n'
        )
        cli.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
        return cli

    def test_returns_subprocess_rc_under_timeout(
        self,
        fake_cli: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the subprocess exits before the deadline, propagate its rc."""
        monkeypatch.setenv("PROGRAMBENCH_FAKE_EXIT", "0")
        rc = _run_programbench_eval(
            tmp_path,
            workers=1,
            branch_workers=1,
            docker_cpus=1,
            image_tag="task",
            force=False,
            timeout=5.0,
        )
        assert rc == 0

        monkeypatch.setenv("PROGRAMBENCH_FAKE_EXIT", "7")
        rc_nonzero = _run_programbench_eval(
            tmp_path,
            workers=1,
            branch_workers=1,
            docker_cpus=1,
            image_tag="task",
            force=False,
            timeout=5.0,
        )
        assert rc_nonzero == 7

    def test_returns_124_on_timeout(
        self,
        fake_cli: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the subprocess wedges past the deadline, kill it and return
        124 (GNU ``timeout`` convention) rather than blocking forever."""
        monkeypatch.setenv("PROGRAMBENCH_FAKE_SLEEP", "10")
        rc = _run_programbench_eval(
            tmp_path,
            workers=1,
            branch_workers=1,
            docker_cpus=1,
            image_tag="task",
            force=False,
            timeout=0.5,
        )
        assert rc == 124

    def test_no_timeout_passes_none_through(
        self,
        fake_cli: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``timeout=None`` (the legacy default for callers that opt out)
        must NOT raise ``TimeoutExpired``: ``subprocess.run`` accepts None
        as "no timeout" and we forward it untouched."""
        monkeypatch.setenv("PROGRAMBENCH_FAKE_SLEEP", "0")
        rc = _run_programbench_eval(
            tmp_path,
            workers=1,
            branch_workers=1,
            docker_cpus=1,
            image_tag="task",
            force=False,
            timeout=None,
        )
        assert rc == 0
