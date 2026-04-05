"""Tests for SkillsBench run_infer module."""

import json
from pathlib import Path

import pytest

from benchmarks.skillsbench.config import INFER_DEFAULTS
from benchmarks.skillsbench.run_infer import (
    convert_harbor_to_eval_output,
    run_harbor_evaluation,
)
from openhands.sdk import LLM


class TestRunHarborEvaluation:
    """Tests for building Harbor invocation arguments."""

    def test_default_dataset_matches_harbor_registry(self) -> None:
        """Test that the default dataset name matches Harbor's published registry."""
        assert INFER_DEFAULTS["dataset"] == "benchflow/skillsbench"

    def test_run_harbor_evaluation_passes_filters_and_limits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test Harbor command includes task filters and n-limit."""
        captured: dict[str, list[str]] = {}

        def fake_run(cmd: list[str], capture_output: bool, text: bool, env: dict):
            captured["cmd"] = cmd
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "ok", "stderr": ""},
            )()

        monkeypatch.setattr("benchmarks.skillsbench.run_infer.subprocess.run", fake_run)

        harbor_output_dir = run_harbor_evaluation(
            llm=LLM(
                model="litellm_proxy/test-model",
                api_key="test-key",
                base_url="https://proxy.example.com",
            ),
            dataset=INFER_DEFAULTS["dataset"],
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
            "-d",
            "benchflow/skillsbench",
            "-a",
            "openhands-sdk",
            "-m",
            "litellm_proxy/test-model",
        ]
        assert "--jobs-dir" in cmd
        assert str(expected_output_dir.resolve()) in cmd
        assert cmd.count("--include-task-name") == 2
        assert "benchflow/task-a" in cmd
        assert "benchflow/task-b" in cmd
        assert cmd[cmd.index("--n-concurrent") + 1] == "2"
        assert cmd[cmd.index("--n-tasks") + 1] == "3"

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

        run_harbor_evaluation(
            llm=LLM(
                model="test-model",
                api_key="my-secret-key",
                base_url="https://my-proxy.example.com",
            ),
            dataset=INFER_DEFAULTS["dataset"],
            output_dir=str(tmp_path),
        )

        assert captured["env"]["LLM_API_KEY"] == "my-secret-key"
        assert captured["env"]["LLM_BASE_URL"] == "https://my-proxy.example.com"


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
