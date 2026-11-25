"""Unit tests for TraceTrial-Bench inference runner utilities."""

from __future__ import annotations

import json
import stat
import tomllib
from pathlib import Path

import pytest
from pydantic import SecretStr

from benchmarks.tracetrialbench.config import INFER_DEFAULTS, TRACETRIAL_PINNED_COMMIT
from benchmarks.tracetrialbench.run_infer import (
    build_tracetrial_model_toml,
    convert_tracetrial_runs_to_eval_output,
    get_parser,
    load_manifest_task_ids,
    resolve_llm_endpoint,
    select_task_ids,
    write_private_model_toml,
)
from openhands.sdk import LLM


def test_resolve_llm_endpoint_litellm_proxy():
    llm = LLM(
        model="litellm_proxy/anthropic/claude-sonnet-4-20250514",
        base_url="https://llm-proxy.eval.all-hands.dev",
        api_key=SecretStr("sk-proxy-secret"),
    )
    provider, model_id, base_url, api_key = resolve_llm_endpoint(llm)
    assert provider == "litellm_proxy"
    assert model_id == "anthropic/claude-sonnet-4-20250514"
    assert base_url == "https://llm-proxy.eval.all-hands.dev/v1"
    assert api_key == "sk-proxy-secret"


def test_resolve_llm_endpoint_provider_prefixed_and_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LLM_API_KEY", "env-key-123")
    llm = LLM(model="deepseek/deepseek-chat", api_key=None)
    provider, model_id, base_url, api_key = resolve_llm_endpoint(llm)
    assert provider == "deepseek"
    assert model_id == "deepseek-chat"
    assert base_url == "https://api.deepseek.com/v1"
    assert api_key == "env-key-123"


def test_resolve_llm_endpoint_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch):
    for var in ("LLM_API_KEY", "LITELLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    llm = LLM(model="openai/gpt-4o", api_key=None)
    with pytest.raises(ValueError, match="LLM API key is required"):
        resolve_llm_endpoint(llm)


def test_build_and_write_private_model_toml(tmp_path: Path):
    agent_llm = LLM(
        model="litellm_proxy/openai/gpt-4o",
        base_url="https://proxy.example.com/v1/",
        api_key=SecretStr('secret-with-"quote"'),
    )
    judge_llm = LLM(
        model="deepseek/deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=SecretStr("judge-key"),
    )
    rendered = build_tracetrial_model_toml(agent_llm, judge_llm)
    dest = write_private_model_toml(rendered, tmp_path / "cfg" / "model.toml")

    mode = stat.S_IMODE(dest.stat().st_mode)
    assert mode == 0o600

    parsed = tomllib.loads(dest.read_text(encoding="utf-8"))
    assert parsed["models"]["eval_agent"]["provider"] == "litellm_proxy"
    assert parsed["models"]["eval_agent"]["model"] == "openai/gpt-4o"
    assert parsed["models"]["eval_agent"]["base_url"] == "https://proxy.example.com/v1"
    assert parsed["models"]["eval_agent"]["api_key"] == 'secret-with-"quote"'
    assert parsed["models"]["eval_judge"]["model"] == "deepseek-chat"


def test_load_and_select_manifest_task_ids(tmp_path: Path):
    manifest = tmp_path / "final.manifest"
    manifest.write_text(
        "# Comment header\n"
        "Task/C1-escalation-export-001\n"
        "Task/E3-firmware-timeout-001\n"
        "T3-reserved-tool-name-001\n\n",
        encoding="utf-8",
    )
    ids = load_manifest_task_ids(manifest)
    assert ids == [
        "C1-escalation-export-001",
        "E3-firmware-timeout-001",
        "T3-reserved-tool-name-001",
    ]

    select_file = tmp_path / "select.txt"
    select_file.write_text(
        "T3-reserved-tool-name-001\nC1-escalation-export-001\n",
        encoding="utf-8",
    )
    selected = select_task_ids(ids, selected_file=str(select_file), n_limit=1)
    assert selected == ["C1-escalation-export-001"]


def test_select_task_ids_unknown_id_raises(tmp_path: Path):
    select_file = tmp_path / "bad.txt"
    select_file.write_text("NON-EXISTENT-001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown task ID"):
        select_task_ids(["C1-escalation-export-001"], selected_file=str(select_file))


def test_convert_tracetrial_runs_to_eval_output(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    run1 = runs_dir / "C1-escalation-export-001__openhands__eval_agent__1111"
    (run1 / "artifacts" / "llm_judge").mkdir(parents=True)
    (run1 / "result.json").write_text(
        json.dumps(
            {
                "task_id": "C1-escalation-export-001",
                "input_query": "Export escalations.",
                "metrics": {
                    "token_usage": {
                        "input_tokens": 1200,
                        "output_tokens": 300,
                        "cached_input_tokens": 800,
                    },
                    "token_cost": {"total_cost": 0.0125},
                },
            }
        ),
        encoding="utf-8",
    )
    (run1 / "artifacts" / "llm_judge" / "combined_score.json").write_text(
        json.dumps(
            {
                "task_id": "C1-escalation-export-001",
                "score": 75.0,
                "pass": False,
                "agent_score": {
                    "points": 50.0,
                    "max_points": 50.0,
                    "passed_items": 4,
                    "total_items": 4,
                },
                "harness_score": {
                    "points": 25.0,
                    "max_points": 50.0,
                    "passed_items": 2,
                    "total_items": 4,
                },
                "attribution": {"attribution": "harness_defect_absorbed"},
            }
        ),
        encoding="utf-8",
    )

    run2 = runs_dir / "E3-firmware-timeout-001__openhands__eval_agent__2222"
    run2.mkdir(parents=True)
    (run2 / "result.json").write_text(
        json.dumps(
            {
                "task_id": "E3-firmware-timeout-001",
                "input_query": "Check rollout.",
                "evaluation": {"error_message": "container timed out"},
                "metrics": {
                    "token_usage": {
                        "input_tokens": "unknown",
                        "output_tokens": "N/A",
                    },
                    "token_cost": {"total_cost": "unknown"},
                },
            }
        ),
        encoding="utf-8",
    )

    output_jsonl = tmp_path / "output.jsonl"
    completed_n, error_n = convert_tracetrial_runs_to_eval_output(
        runs_dir,
        output_jsonl,
        expected_task_ids=[
            "C1-escalation-export-001",
            "E3-firmware-timeout-001",
            "T3-reserved-tool-name-001",
        ],
    )
    assert completed_n == 1
    assert error_n == 2

    rows = [
        json.loads(line)
        for line in output_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 3

    first = rows[0]
    assert first["instance_id"] == "C1-escalation-export-001"
    assert first["error"] is None
    assert first["test_result"]["agent_pass"] is True
    assert first["test_result"]["harness_pass"] is False
    assert first["test_result"]["clean_pass"] is False
    assert first["test_result"]["agent_score_pct"] == 100.0
    assert first["test_result"]["harness_score_pct"] == 50.0
    assert first["test_result"]["overall_score"] == 75.0
    assert first["test_result"]["attribution"] == "harness_defect_absorbed"
    assert first["metrics"]["accumulated_cost"] == 0.0125
    assert first["metrics"]["total_prompt_tokens"] == 1200

    assert rows[1]["instance_id"] == "E3-firmware-timeout-001"
    assert rows[1]["error"] == "container timed out"
    assert rows[2]["instance_id"] == "T3-reserved-tool-name-001"
    assert "No TraceTrial-Bench result directory" in rows[2]["error"]


def test_cli_parser_defaults():
    parser = get_parser()
    args = parser.parse_args([".llm_config/eval.json"])
    assert args.llm_config_path == ".llm_config/eval.json"
    assert args.manifest == INFER_DEFAULTS["manifest"]
    assert args.num_workers == INFER_DEFAULTS["num_workers"]
    assert args.use_vendor_sdk is True
    assert args.tracetrial_commit == TRACETRIAL_PINNED_COMMIT
