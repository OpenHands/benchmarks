"""Tests for SkillsBench run_infer module."""

import json
from pathlib import Path

import pytest
import yaml

from benchmarks.skillsbench.config import BENCHFLOW_DEFAULTS, INFER_DEFAULTS
from benchmarks.skillsbench.run_infer import (
    _build_benchflow_agent_env,
    convert_benchflow_to_eval_output,
    run_benchflow_job,
)
from openhands.sdk import LLM


class TestRunBenchflowJob:
    """Tests for building benchflow job invocation arguments."""

    def test_default_dataset_matches_benchflow_registry(self) -> None:
        """Test that the default dataset name matches benchflow's published registry."""
        assert INFER_DEFAULTS["dataset"] == "benchflow/skillsbench"

    def test_default_agent_is_openhands(self) -> None:
        """Test that the default agent is openhands."""
        assert BENCHFLOW_DEFAULTS["agent_name"] == "openhands"

    def test_run_benchflow_job_passes_model_and_concurrency(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test benchflow job command writes the expected YAML config."""
        captured_cmd: list[str] = []
        captured_env: dict[str, str] = {}
        captured_config: dict = {}

        # Force legacy benchflow binary path so the command format is deterministic
        monkeypatch.setattr(
            "benchmarks.skillsbench.run_infer.shutil.which",
            lambda name: "/usr/local/bin/benchflow" if name == "benchflow" else None,
        )

        def fake_run(cmd: list[str], capture_output: bool, text: bool, env: dict):
            captured_cmd[:] = cmd
            captured_env.clear()
            captured_env.update(env)
            with open(cmd[3]) as f:
                captured_config.update(yaml.safe_load(f))
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "Score: 1/1 (100%)", "stderr": ""},
            )()

        monkeypatch.setattr("benchmarks.skillsbench.run_infer.subprocess.run", fake_run)

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        jobs_dir = tmp_path / "jobs"

        run_benchflow_job(
            llm=LLM(
                model="anthropic/claude-sonnet-4-5",
                api_key="test-key",
                base_url="https://proxy.example.com",
            ),
            tasks_dir=tasks_dir,
            jobs_dir=jobs_dir,
            num_workers=4,
        )

        cmd = captured_cmd
        assert cmd[0] == "/usr/local/bin/benchflow"
        assert cmd[1] == "job"
        assert cmd[2] == "--config"
        assert captured_config["tasks_dir"] == str(tasks_dir)
        assert captured_config["jobs_dir"] == str(jobs_dir.resolve())
        assert captured_config["agent"] == "openhands"
        assert captured_config["model"] == "anthropic/claude-sonnet-4-5"
        assert captured_config["concurrency"] == 4
        assert captured_config["sandbox_user"] is None

    def test_llm_credentials_passed_via_subprocess_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that LLM credentials are passed via subprocess env and YAML."""
        captured_cmd: list[str] = []
        captured_env: dict[str, str] = {}
        captured_config: dict = {}

        # Force legacy benchflow binary path so the command format is deterministic
        monkeypatch.setattr(
            "benchmarks.skillsbench.run_infer.shutil.which",
            lambda name: "/usr/local/bin/benchflow" if name == "benchflow" else None,
        )

        def fake_run(cmd: list[str], capture_output: bool, text: bool, env: dict):
            captured_cmd[:] = cmd
            captured_env.clear()
            captured_env.update(env)
            with open(cmd[3]) as f:
                captured_config.update(yaml.safe_load(f))
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "ok", "stderr": ""},
            )()

        monkeypatch.setattr("benchmarks.skillsbench.run_infer.subprocess.run", fake_run)

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        run_benchflow_job(
            llm=LLM(
                model="test-model",
                api_key="my-secret-key",
                base_url="https://my-proxy.example.com",
            ),
            tasks_dir=tasks_dir,
            jobs_dir=tmp_path / "jobs",
        )

        # Credentials in subprocess env
        assert captured_env["LLM_API_KEY"] == "my-secret-key"
        assert captured_env["LLM_BASE_URL"] == "https://my-proxy.example.com"
        assert "--ae" not in captured_cmd
        assert captured_config["agent_env"]["LLM_API_KEY"] == "my-secret-key"
        assert (
            captured_config["agent_env"]["LLM_BASE_URL"]
            == "https://my-proxy.example.com"
        )

    def test_direct_gemini_model_sets_provider_env_vars(self) -> None:
        """Direct provider models need provider-specific env vars."""
        env = _build_benchflow_agent_env(
            LLM(
                model="gemini/gemini-3.1-flash-lite-preview",
                api_key="gemini-test-key",
            )
        )

        assert env["LLM_API_KEY"] == "gemini-test-key"
        assert env["GEMINI_API_KEY"] == "gemini-test-key"
        assert env["GOOGLE_API_KEY"] == "gemini-test-key"

    def test_proxy_model_does_not_set_provider_env_vars(self) -> None:
        """LiteLLM proxy configs should keep using generic LLM_* vars only."""
        env = _build_benchflow_agent_env(
            LLM(
                model="litellm_proxy/anthropic/claude-sonnet-4-20250514",
                api_key="proxy-key",
                base_url="https://proxy.example.com",
            )
        )

        assert env["LLM_API_KEY"] == "proxy-key"
        assert env["LLM_BASE_URL"] == "https://proxy.example.com"
        assert "ANTHROPIC_API_KEY" not in env
        assert "ANTHROPIC_BASE_URL" not in env


class TestConvertBenchflowToEvalOutput:
    """Tests for convert_benchflow_to_eval_output function."""

    def _create_benchflow_structure(
        self, tmp_path: Path, tasks: list[tuple[str, dict]]
    ) -> Path:
        """Create a mock benchflow jobs directory structure.

        benchflow writes: jobs_dir/TASK_NAME/trial-0/result.json
        """
        jobs_dir = tmp_path / "jobs"
        for task_name, result in tasks:
            trial_dir = jobs_dir / task_name / "trial-0"
            trial_dir.mkdir(parents=True)
            (trial_dir / "result.json").write_text(json.dumps(result))
        return jobs_dir

    def _create_benchflow_timestamped_job(
        self, tmp_path: Path, tasks: list[tuple[str, dict]]
    ) -> Path:
        """Create a mock benchflow 0.3.0 jobs directory structure.

        benchflow writes: jobs/TIMESTAMP/TASK_NAME__RUNID/result.json
        """
        jobs_dir = tmp_path / "jobs"
        job_dir = jobs_dir / "2026-04-21__23-12-35"
        job_dir.mkdir(parents=True)
        (jobs_dir / "summary.json").write_text(json.dumps({"total": len(tasks)}))
        for task_name, result in tasks:
            trial_dir = job_dir / f"{task_name}__abc123"
            trial_dir.mkdir(parents=True)
            (trial_dir / "result.json").write_text(json.dumps(result))
        return jobs_dir

    def test_successful_trial_parsing(self, tmp_path: Path) -> None:
        """Test successful parsing of a benchflow trial result.

        benchflow 0.3.0 does not write cost/token fields to result.json.
        Metrics are read from agent/trajectory.json (harbor-format agent)
        or parsed from agent/openhands.txt (ACP agent stdout).
        """
        trial_result = {
            "task_name": "benchflow/weighted-gdp-calc",
            "rewards": {"reward": 1.0},
            "error": None,
        }

        jobs_dir = self._create_benchflow_structure(
            tmp_path, [("weighted-gdp-calc", trial_result)]
        )
        # Write agent/trajectory.json with final_metrics (harbor-format agent output).
        # agent/ sits next to result.json, inside the trial-0 subdirectory.
        trial_dir = jobs_dir / "weighted-gdp-calc" / "trial-0"
        agent_dir = trial_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "trajectory.json").write_text(
            json.dumps(
                {
                    "final_metrics": {
                        "total_prompt_tokens": 1000,
                        "total_completion_tokens": 200,
                        "total_cost_usd": 0.05,
                    }
                }
            )
        )
        output_file = tmp_path / "output.jsonl"

        convert_benchflow_to_eval_output(jobs_dir, output_file)

        assert output_file.exists()
        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 1
        assert entries[0]["instance_id"] == "benchflow/weighted-gdp-calc"
        assert entries[0]["test_result"]["passed"] is True
        assert entries[0]["metrics"]["total_cost_usd"] == 0.05
        assert entries[0]["metrics"]["total_prompt_tokens"] == 1000
        assert entries[0]["metrics"]["total_completion_tokens"] == 200

    def test_metrics_from_acp_agent_log(self, tmp_path: Path) -> None:
        """Test that metrics are extracted from agent/openhands.txt (ACP agent)."""
        trial_result = {
            "task_name": "benchflow/acp-task",
            "rewards": {"reward": 1.0},
            "error": None,
        }
        jobs_dir = self._create_benchflow_timestamped_job(
            tmp_path, [("acp-task", trial_result)]
        )
        # Write agent/openhands.txt simulating openhands ACP stdout
        trial_dir = jobs_dir / "2026-04-21__23-12-35" / "acp-task__abc123"
        agent_dir = trial_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "openhands.txt").write_text(
            "OpenHands SDK v1.16.0\n"
            "Tokens: ↑ input 404.21K • cache hit 70.47% •  reasoning 579 • ↓ output 7.83K • $0.0487\n"
            "Total cost: $0.0487\n"
        )
        output_file = tmp_path / "output.jsonl"
        convert_benchflow_to_eval_output(jobs_dir, output_file)

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 1
        assert entries[0]["metrics"]["total_cost_usd"] == pytest.approx(0.0487)
        assert entries[0]["metrics"]["total_prompt_tokens"] == 404210
        assert entries[0]["metrics"]["total_completion_tokens"] == 7830

    def test_failed_trial(self, tmp_path: Path) -> None:
        """Test parsing of a trial with reward 0."""
        trial_result = {
            "task_name": "benchflow/task-1",
            "rewards": {"reward": 0.0},
            "error": None,
        }

        jobs_dir = self._create_benchflow_structure(
            tmp_path, [("task-1", trial_result)]
        )
        output_file = tmp_path / "output.jsonl"
        convert_benchflow_to_eval_output(jobs_dir, output_file)

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert entries[0]["test_result"]["passed"] is False
        assert entries[0]["metrics"]["total_cost_usd"] == 0.0

    def test_trial_with_error(self, tmp_path: Path) -> None:
        """Test that errored trials are written as error entries."""
        trial_result = {
            "task_name": "benchflow/error-task",
            "rewards": {},
            "error": "LLM_API_KEY not set",
        }

        jobs_dir = self._create_benchflow_structure(
            tmp_path, [("error-task", trial_result)]
        )
        output_file = tmp_path / "output.jsonl"
        convert_benchflow_to_eval_output(jobs_dir, output_file)

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 1
        assert entries[0]["instance_id"] == "benchflow/error-task"
        assert entries[0]["error"] is not None
        assert entries[0]["test_result"] == {}

    def test_missing_jobs_directory(self, tmp_path: Path) -> None:
        """Test handling when jobs directory is empty."""
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()

        with pytest.raises(RuntimeError, match="No task directories found"):
            convert_benchflow_to_eval_output(jobs_dir, tmp_path / "output.jsonl")

    def test_task_id_filtering(self, tmp_path: Path) -> None:
        """Test that only specified task IDs are converted."""
        trials = [
            (
                "task-a",
                {
                    "task_name": "benchflow/task-a",
                    "rewards": {"reward": 1.0},
                    "error": None,
                },
            ),
            (
                "task-b",
                {
                    "task_name": "benchflow/task-b",
                    "rewards": {"reward": 0.0},
                    "error": None,
                },
            ),
        ]
        jobs_dir = self._create_benchflow_structure(tmp_path, trials)
        output_file = tmp_path / "output.jsonl"

        convert_benchflow_to_eval_output(
            jobs_dir, output_file, task_ids=["benchflow/task-a"]
        )

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 1
        assert entries[0]["instance_id"] == "benchflow/task-a"

    def test_task_name_normalised_to_benchflow_prefix(self, tmp_path: Path) -> None:
        """Test that task names without prefix get benchflow/ prepended."""
        trial_result = {
            "task_name": "weighted-gdp-calc",  # no benchflow/ prefix
            "rewards": {"reward": 1.0},
            "error": None,
        }
        jobs_dir = self._create_benchflow_structure(
            tmp_path, [("weighted-gdp-calc", trial_result)]
        )
        output_file = tmp_path / "output.jsonl"
        convert_benchflow_to_eval_output(jobs_dir, output_file)

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert entries[0]["instance_id"] == "benchflow/weighted-gdp-calc"

    def test_timestamped_job_directory_is_processed(self, tmp_path: Path) -> None:
        """Test benchflow 0.3.0 timestamped jobs directory layout."""
        trial_result = {
            "task_name": "weighted-gdp-calc",
            "rewards": {"reward": 1.0},
            "error": None,
            "n_input_tokens": 42,
            "n_output_tokens": 7,
            "cost_usd": 0.01,
        }

        jobs_dir = self._create_benchflow_timestamped_job(
            tmp_path, [("weighted-gdp-calc", trial_result)]
        )
        output_file = tmp_path / "output.jsonl"

        convert_benchflow_to_eval_output(jobs_dir, output_file)

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 1
        assert entries[0]["instance_id"] == "benchflow/weighted-gdp-calc"
        assert entries[0]["test_result"]["passed"] is True

    def test_task_id_filter_matches_timestamped_trial_dir(self, tmp_path: Path) -> None:
        """Test filtering strips the run suffix from trial directory names."""
        jobs_dir = self._create_benchflow_timestamped_job(
            tmp_path,
            [
                (
                    "task-a",
                    {
                        "task_name": "task-a",
                        "rewards": {"reward": 1.0},
                        "error": None,
                    },
                ),
                (
                    "task-b",
                    {
                        "task_name": "task-b",
                        "rewards": {"reward": 0.0},
                        "error": None,
                    },
                ),
            ],
        )
        output_file = tmp_path / "output.jsonl"

        convert_benchflow_to_eval_output(
            jobs_dir, output_file, task_ids=["benchflow/task-a"]
        )

        with open(output_file) as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 1
        assert entries[0]["instance_id"] == "benchflow/task-a"
