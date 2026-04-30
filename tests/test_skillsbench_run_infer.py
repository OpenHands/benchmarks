"""Tests for SkillsBench run_infer module."""

import json
from pathlib import Path

import pytest

from benchmarks.skillsbench.config import INFER_DEFAULTS
from benchmarks.skillsbench.run_infer import (
    convert_harbor_to_eval_output,
    ensure_skillsbench_tasks,
    resolve_skillsbench_dataset,
    run_harbor_evaluation,
)
from openhands.sdk import LLM


class TestDatasetSync:
    """Tests for syncing the local SkillsBench task snapshot."""

    def test_ensure_skillsbench_tasks_reuses_matching_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that an up-to-date cached tasks directory is reused."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "task-a").mkdir()
        metadata_path = tmp_path / "source.json"
        metadata_path.write_text(json.dumps({"commit_hash": "abc123"}))

        monkeypatch.setattr(
            "benchmarks.skillsbench.run_infer.get_skillsbench_main_commit",
            lambda repo_url, branch: "abc123",
        )

        called = False

        def fake_download(**kwargs) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(
            "benchmarks.skillsbench.run_infer.download_skillsbench_tasks",
            fake_download,
        )

        resolved = ensure_skillsbench_tasks(
            tasks_dir=tasks_dir,
            metadata_path=metadata_path,
        )

        assert resolved == tasks_dir
        assert called is False

    def test_ensure_skillsbench_tasks_refreshes_stale_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that a stale cached commit triggers a redownload."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        metadata_path = tmp_path / "source.json"
        metadata_path.write_text(json.dumps({"commit_hash": "old-commit"}))

        monkeypatch.setattr(
            "benchmarks.skillsbench.run_infer.get_skillsbench_main_commit",
            lambda repo_url, branch: "new-commit",
        )

        captured: dict[str, str] = {}

        def fake_download(
            *,
            commit_hash: str,
            tasks_dir: Path,
            metadata_path: Path,
            repo_url: str,
            branch: str,
        ) -> None:
            captured["commit_hash"] = commit_hash
            captured["tasks_dir"] = str(tasks_dir)
            captured["metadata_path"] = str(metadata_path)
            tasks_dir.mkdir(exist_ok=True)

        monkeypatch.setattr(
            "benchmarks.skillsbench.run_infer.download_skillsbench_tasks",
            fake_download,
        )

        ensure_skillsbench_tasks(
            tasks_dir=tasks_dir,
            metadata_path=metadata_path,
        )

        assert captured["commit_hash"] == "new-commit"
        assert captured["tasks_dir"] == str(tasks_dir)
        assert captured["metadata_path"] == str(metadata_path)

    def test_ensure_skillsbench_tasks_uses_cache_if_remote_check_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that a usable cache is kept when the upstream HEAD check fails."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "task-a").mkdir()
        metadata_path = tmp_path / "source.json"
        metadata_path.write_text(json.dumps({"commit_hash": "cached-commit"}))

        def fake_head(repo_url: str, branch: str) -> str:
            raise RuntimeError("network unavailable")

        monkeypatch.setattr(
            "benchmarks.skillsbench.run_infer.get_skillsbench_main_commit",
            fake_head,
        )

        resolved = ensure_skillsbench_tasks(
            tasks_dir=tasks_dir,
            metadata_path=metadata_path,
        )

        assert resolved == tasks_dir

    def test_resolve_skillsbench_dataset_maps_aliases_to_local_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test SkillsBench dataset aliases resolve to the local Harbor dataset."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        monkeypatch.setattr(
            "benchmarks.skillsbench.run_infer.ensure_skillsbench_tasks",
            lambda: tasks_dir,
        )

        resolved_dataset, dataset_is_path = resolve_skillsbench_dataset(
            "benchflow/skillsbench@1.0"
        )

        assert resolved_dataset == str(tasks_dir.resolve())
        assert dataset_is_path is True


class TestRunHarborEvaluation:
    """Tests for building Harbor invocation arguments."""

    def test_run_harbor_evaluation_passes_filters_and_limits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test Harbor command normalizes local task ids and includes main flags."""
        captured: dict[str, list[str]] = {}

        def fake_run(cmd: list[str], capture_output: bool, text: bool, env: dict):
            captured["cmd"] = cmd
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "ok", "stderr": ""},
            )()

        monkeypatch.setattr("benchmarks.skillsbench.run_infer.subprocess.run", fake_run)
        monkeypatch.setattr(
            "benchmarks.skillsbench.run_infer._get_supported_task_filter_flag",
            lambda harbor_exe: "--include-task-name",
        )
        monkeypatch.setattr(
            "benchmarks.skillsbench.run_infer._get_supported_agent_name",
            lambda harbor_exe: "openhands",
        )

        harbor_output_dir = run_harbor_evaluation(
            llm=LLM(
                model="litellm_proxy/test-model",
                api_key="test-key",
                base_url="https://proxy.example.com",
            ),
            dataset=str(tmp_path / "tasks"),
            dataset_is_path=True,
            output_dir=str(tmp_path),
            num_workers=2,
            task_ids=["benchflow/task-a", "benchflow/task-b"],
            n_limit=3,
        )

        expected_output_dir = tmp_path / "harbor_output"
        assert harbor_output_dir == expected_output_dir

        cmd = captured["cmd"]
        assert cmd[:8] == [
            "harbor",
            "run",
            "--path",
            str(tmp_path / "tasks"),
            "-a",
            "openhands",
            "-m",
            "litellm_proxy/test-model",
        ]
        assert "--jobs-dir" in cmd
        assert str(expected_output_dir.resolve()) in cmd
        assert cmd.count("--include-task-name") == 2
        assert "task-a" in cmd
        assert "task-b" in cmd
        assert "benchflow/task-a" not in cmd
        assert "--ae" not in cmd
        assert cmd[cmd.index("--n-concurrent") + 1] == "2"
        assert cmd[cmd.index("--n-tasks") + 1] == "3"

    def test_run_harbor_evaluation_retries_with_legacy_task_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test Harbor falls back to --include-task-name when --task-name fails."""
        captured_cmds: list[list[str]] = []

        def fake_run(cmd: list[str], capture_output: bool, text: bool, env: dict):
            captured_cmds.append(cmd)
            if "--task-name" in cmd:
                return type(
                    "Completed",
                    (),
                    {
                        "returncode": 2,
                        "stdout": "",
                        "stderr": "No such option: --task-name",
                    },
                )()
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "ok", "stderr": ""},
            )()

        monkeypatch.setattr("benchmarks.skillsbench.run_infer.subprocess.run", fake_run)
        monkeypatch.setattr(
            "benchmarks.skillsbench.run_infer._get_supported_task_filter_flag",
            lambda harbor_exe: "--task-name",
        )
        monkeypatch.setattr(
            "benchmarks.skillsbench.run_infer._get_supported_agent_name",
            lambda harbor_exe: "openhands",
        )

        run_harbor_evaluation(
            llm=LLM(model="test-model"),
            dataset=str(tmp_path / "tasks"),
            dataset_is_path=True,
            output_dir=str(tmp_path),
            task_ids=["benchflow/task-a"],
        )

        assert len(captured_cmds) == 2
        assert "--task-name" in captured_cmds[0]
        assert "--include-task-name" in captured_cmds[1]

    def test_llm_credentials_passed_via_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that LLM credentials are passed via subprocess env, not --ae flags."""
        captured: dict = {}

        def fake_run(cmd: list[str], capture_output: bool, text: bool, env: dict):
            captured["cmd"] = cmd
            captured["env"] = env
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "ok", "stderr": ""},
            )()

        monkeypatch.setattr("benchmarks.skillsbench.run_infer.subprocess.run", fake_run)
        monkeypatch.setattr(
            "benchmarks.skillsbench.run_infer._get_supported_task_filter_flag",
            lambda harbor_exe: "--include-task-name",
        )
        monkeypatch.setattr(
            "benchmarks.skillsbench.run_infer._get_supported_agent_name",
            lambda harbor_exe: "openhands",
        )

        run_harbor_evaluation(
            llm=LLM(
                model="test-model",
                api_key="my-secret-key",
                base_url="https://my-proxy.example.com",
            ),
            dataset=INFER_DEFAULTS["dataset"],
            dataset_is_path=False,
            output_dir=str(tmp_path),
        )

        assert captured["env"]["LLM_API_KEY"] == "my-secret-key"
        assert captured["env"]["LLM_BASE_URL"] == "https://my-proxy.example.com"
        assert "--ae" not in captured["cmd"]


class TestConvertHarborToEvalOutput:
    """Tests for convert_harbor_to_eval_output function."""

    def _create_harbor_structure(
        self, tmp_path: Path, trials: list[tuple[str, dict]]
    ) -> Path:
        """Create a mock Harbor output structure."""
        harbor_dir = tmp_path / "harbor_output"
        job_dir = harbor_dir / "2026-01-01__00-00-00"
        job_dir.mkdir(parents=True)
        (job_dir / "result.json").write_text(json.dumps({"id": "test-job"}))

        for trial_name, trial_result in trials:
            trial_dir = job_dir / trial_name
            trial_dir.mkdir()
            (trial_dir / "result.json").write_text(json.dumps(trial_result))

        return harbor_dir

    def test_successful_trial_parsing(self, tmp_path: Path) -> None:
        """Test successful parsing of harbor trial result."""
        trial_result = {
            "task_name": "benchflow/weighted-gdp-calc",
            "trial_name": "weighted-gdp-calc__abc123",
            "trial_uri": "file:///path/to/trial",
            "agent_result": {
                "n_input_tokens": 1000,
                "n_output_tokens": 200,
                "cost_usd": 0.05,
            },
            "verifier_result": {"rewards": {"reward": 1.0}},
            "exception_info": None,
        }

        harbor_dir = self._create_harbor_structure(
            tmp_path, [("weighted-gdp-calc__abc123", trial_result)]
        )
        output_file = tmp_path / "output.jsonl"

        convert_harbor_to_eval_output(harbor_dir, output_file)

        assert output_file.exists()
        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 1
        assert entries[0]["instance_id"] == "benchflow/weighted-gdp-calc"
        assert entries[0]["test_result"]["passed"] is True
        assert entries[0]["metrics"]["total_cost_usd"] == 0.05

    def test_local_trial_names_are_normalized_to_canonical_instance_ids(
        self, tmp_path: Path
    ) -> None:
        """Test local Harbor task names without namespace keep benchflow ids."""
        trial_result = {
            "task_name": "weighted-gdp-calc",
            "trial_name": "weighted-gdp-calc__abc123",
            "trial_uri": "file:///path/to/trial",
            "agent_result": {
                "n_input_tokens": 1000,
                "n_output_tokens": 200,
                "cost_usd": 0.05,
            },
            "verifier_result": {"rewards": {"reward": 1.0}},
            "exception_info": None,
        }

        harbor_dir = self._create_harbor_structure(
            tmp_path, [("weighted-gdp-calc__abc123", trial_result)]
        )
        output_file = tmp_path / "output.jsonl"

        convert_harbor_to_eval_output(harbor_dir, output_file)

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert entries[0]["instance_id"] == "benchflow/weighted-gdp-calc"

    def test_failed_trial(self, tmp_path: Path) -> None:
        """Test parsing of a trial with reward 0."""
        trial_result = {
            "task_name": "benchflow/task-1",
            "trial_name": "task-1__xyz",
            "agent_result": {
                "n_input_tokens": None,
                "n_output_tokens": None,
                "cost_usd": None,
            },
            "verifier_result": {"rewards": {"reward": 0.0}},
            "exception_info": None,
        }

        harbor_dir = self._create_harbor_structure(
            tmp_path, [("task-1__xyz", trial_result)]
        )
        output_file = tmp_path / "output.jsonl"
        convert_harbor_to_eval_output(harbor_dir, output_file)

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert entries[0]["test_result"]["passed"] is False
        assert entries[0]["metrics"]["total_cost_usd"] == 0.0

    def test_trial_with_exception(self, tmp_path: Path) -> None:
        """Test that exception trials are written as error entries."""
        trial_result = {
            "task_name": "benchflow/error-task",
            "trial_name": "error-task__err",
            "agent_result": {},
            "verifier_result": {},
            "exception_info": {"type": "ValueError", "message": "LLM_API_KEY not set"},
        }

        harbor_dir = self._create_harbor_structure(
            tmp_path, [("error-task__err", trial_result)]
        )
        output_file = tmp_path / "output.jsonl"
        convert_harbor_to_eval_output(harbor_dir, output_file)

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 1
        assert entries[0]["instance_id"] == "benchflow/error-task"
        assert entries[0]["error"] is not None
        assert entries[0]["test_result"] == {}

    def test_missing_job_directory(self, tmp_path: Path) -> None:
        """Test handling when no job directory exists."""
        harbor_dir = tmp_path / "harbor_output"
        harbor_dir.mkdir()

        with pytest.raises(RuntimeError, match="No harbor job directory found"):
            convert_harbor_to_eval_output(harbor_dir, tmp_path / "output.jsonl")

    def test_empty_job_directory(self, tmp_path: Path) -> None:
        """Test handling of harbor job dir with no trial subdirs."""
        harbor_dir = tmp_path / "harbor_output"
        job_dir = harbor_dir / "2026-01-01__00-00-00"
        job_dir.mkdir(parents=True)
        (job_dir / "result.json").write_text(json.dumps({"id": "test"}))

        with pytest.raises(RuntimeError, match="No trial result files found"):
            convert_harbor_to_eval_output(harbor_dir, tmp_path / "output.jsonl")
