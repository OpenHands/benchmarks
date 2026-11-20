#!/usr/bin/env python3
"""TraceTrial-Bench inference and scoring runner for OpenHands.

Executes TraceTrial-Bench (https://github.com/TraceLite-AI/TraceTrial-Bench,
arXiv:2610.32285) against the containerized OpenHands harness, optionally
overlaying the local ``vendor/software-agent-sdk`` workspace packages so CI
and developer runs evaluate the exact checked-out SDK revision.

Usage:
    uv run tracetrialbench-infer .llm_config/eval.json --n-limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from benchmarks.tracetrialbench.config import (
    DATASET_NAME,
    INFER_DEFAULTS,
    TRACETRIAL_PINNED_COMMIT,
    TRACETRIAL_REPO_URL,
)
from benchmarks.utils.constants import OUTPUT_FILENAME
from benchmarks.utils.evaluation_utils import construct_eval_output_dir
from benchmarks.utils.llm_config import load_llm_config
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import LLM, get_logger


logger = get_logger(__name__)

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent.parent
VENDOR_SDK_DIR = REPO_ROOT / "vendor" / "software-agent-sdk"
CACHE_DIR = BENCHMARK_DIR / "data"
REPO_CACHE_DIR = CACHE_DIR / "TraceTrial-Bench"
VENV_CACHE_DIR = CACHE_DIR / ".venv"
SOURCE_METADATA_PATH = CACHE_DIR / "source.json"

DEFAULT_PROVIDER_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def _extract_secret(secret: SecretStr | str | None) -> str:
    if isinstance(secret, SecretStr):
        return secret.get_secret_value()
    if isinstance(secret, str):
        return secret
    return ""


def resolve_llm_endpoint(llm: LLM) -> tuple[str, str, str, str]:
    """Extract ``(provider, model_id, base_url, api_key)`` from an OpenHands ``LLM``.

    Handles ``litellm_proxy/<vendor>/<model>``, ``<provider>/<model>``, and bare
    model names while normalizing OpenAI-compatible ``/v1`` base URLs for the
    TraceTrial-Bench model registry.
    """
    raw_model = llm.model.strip()
    if not raw_model:
        raise ValueError("LLM configuration has an empty model identifier")

    if raw_model.startswith("litellm_proxy/"):
        provider = "litellm_proxy"
        model_id = raw_model[len("litellm_proxy/") :]
    elif "/" in raw_model:
        provider, _, model_id = raw_model.partition("/")
    else:
        provider = "openai"
        model_id = raw_model

    if not provider or not model_id:
        raise ValueError(f"Invalid LLM model identifier: {llm.model!r}")

    raw_base_url = (llm.base_url or "").strip().rstrip("/")
    if not raw_base_url:
        raw_base_url = DEFAULT_PROVIDER_URLS.get(
            provider.lower(), "https://api.openai.com/v1"
        )
    elif provider == "litellm_proxy" and not raw_base_url.endswith("/v1"):
        raw_base_url = f"{raw_base_url}/v1"

    api_key = _extract_secret(llm.api_key).strip()
    if not api_key:
        api_key = (
            os.environ.get("LLM_API_KEY")
            or os.environ.get("LITELLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or ""
        ).strip()
    if not api_key:
        raise ValueError(
            "LLM API key is required for TraceTrial-Bench; set 'api_key' in the "
            "LLM JSON config or export LLM_API_KEY."
        )

    return provider, model_id, raw_base_url, api_key


def _toml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_tracetrial_model_toml(
    agent_llm: LLM,
    judge_llm: LLM | None = None,
    *,
    agent_alias: str = "eval_agent",
    judge_alias: str = "eval_judge",
) -> str:
    """Render a private ``models.toml`` document for ``tracetrial_bench``."""
    profiles: list[tuple[str, LLM]] = [(agent_alias, agent_llm)]
    if judge_llm is not None:
        profiles.append((judge_alias, judge_llm))

    blocks: list[str] = []
    for alias, llm in profiles:
        provider, model_id, base_url, api_key = resolve_llm_endpoint(llm)
        blocks.append(
            "\n".join(
                [
                    f"[models.{alias}]",
                    f"provider = {_toml_quote(provider)}",
                    f"model = {_toml_quote(model_id)}",
                    f"base_url = {_toml_quote(base_url)}",
                    'api_mode = "chat_completions"',
                    f"api_key = {_toml_quote(api_key)}",
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def write_private_model_toml(content: str, destination: Path) -> Path:
    """Write ``models.toml`` with strict ``0600`` permissions required by upstream."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.chmod(destination, 0o600)
    return destination


def load_manifest_task_ids(manifest_path: Path) -> list[str]:
    """Read ordered task IDs from a TraceTrial-Bench ``.manifest`` file."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Task manifest not found: {manifest_path}")
    task_ids: list[str] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            task_ids.append(Path(line).name)
    if not task_ids:
        raise ValueError(f"Task manifest {manifest_path} contains no tasks")
    return task_ids


def select_task_ids(
    manifest_ids: list[str],
    *,
    selected_file: str | None = None,
    n_limit: int = 0,
) -> list[str]:
    """Filter manifest task IDs by ``--select`` file and ``--n-limit``."""
    filtered = list(manifest_ids)
    if selected_file:
        sel_path = Path(selected_file)
        if not sel_path.is_file():
            raise FileNotFoundError(f"--select file does not exist: {sel_path}")
        wanted = {
            line.strip()
            for line in sel_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        if not wanted:
            raise ValueError(f"--select file {selected_file!r} is empty")
        known = set(filtered)
        missing = sorted(wanted - known)
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"--select listed {len(missing)} unknown task ID(s): {preview}"
            )
        filtered = [tid for tid in filtered if tid in wanted]
    if n_limit > 0:
        filtered = filtered[:n_limit]
    return filtered


def _run_checked(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    error_prefix: str,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{error_prefix}: {detail}")
    return completed


def _cached_commit(metadata_path: Path = SOURCE_METADATA_PATH) -> str | None:
    if not metadata_path.is_file():
        return None
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    commit = data.get("commit")
    return commit if isinstance(commit, str) and commit else None


def ensure_tracetrial_checkout(
    *,
    repo_url: str = TRACETRIAL_REPO_URL,
    commit: str = TRACETRIAL_PINNED_COMMIT,
    repo_dir: Path = REPO_CACHE_DIR,
    venv_dir: Path = VENV_CACHE_DIR,
    metadata_path: Path = SOURCE_METADATA_PATH,
) -> Path:
    """Clone or reuse the pinned TraceTrial-Bench checkout and CLI venv."""
    cli_bin = venv_dir / "bin" / "tracetrial_bench"
    if (
        _cached_commit(metadata_path) == commit
        and (repo_dir / "Task" / "final.manifest").is_file()
        and cli_bin.is_file()
    ):
        logger.info("Reusing cached TraceTrial-Bench checkout at %s", commit[:12])
        return cli_bin

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    logger.info("Cloning %s at %s into %s", repo_url, commit[:12], repo_dir)
    _run_checked(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            repo_url,
            str(repo_dir),
        ],
        error_prefix="Failed to clone TraceTrial-Bench repository",
    )
    _run_checked(
        ["git", "checkout", "--detach", commit],
        cwd=repo_dir,
        error_prefix=f"Failed to checkout TraceTrial-Bench commit {commit}",
    )

    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    _run_checked(
        [sys.executable, "-m", "venv", str(venv_dir)],
        error_prefix="Failed to create TraceTrial-Bench virtualenv",
    )
    _run_checked(
        [str(venv_dir / "bin" / "pip"), "install", "--quiet", "-e", str(repo_dir)],
        error_prefix="Failed to install TraceTrial-Bench package",
    )

    metadata_path.write_text(
        json.dumps({"repo_url": repo_url, "commit": commit}, indent=2) + "\n",
        encoding="utf-8",
    )
    return cli_bin


def _find_combined_score(run_dir: Path) -> dict[str, Any] | None:
    candidates = [
        run_dir / "artifacts" / "llm_judge" / "combined_score.json",
        run_dir / "combined_score.json",
    ]
    for path in candidates:
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    return loaded
            except (OSError, json.JSONDecodeError):
                continue
    return None


def _coerce_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
    return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _layer_pct(layer: Any) -> tuple[float, bool, int, int]:
    if not isinstance(layer, dict):
        return 0.0, False, 0, 0
    passed = _coerce_int(layer.get("passed_items"), 0)
    total = _coerce_int(layer.get("total_items"), 0)
    points = _coerce_float(layer.get("points"), 0.0)
    max_points = _coerce_float(layer.get("max_points"), 50.0)
    pct = (points / max_points * 100.0) if max_points > 0 else 0.0
    all_passed = total > 0 and passed == total
    return round(pct, 4), all_passed, passed, total


def convert_tracetrial_runs_to_eval_output(
    runs_dir: Path,
    eval_output_path: Path,
    *,
    expected_task_ids: list[str] | None = None,
) -> tuple[int, int]:
    """Convert ``tracetrial_bench`` run directories into OpenHands ``output.jsonl``.

    Returns ``(completed_count, error_count)``.
    """
    result_files = sorted(runs_dir.glob("*/result.json"))
    latest_by_task: dict[str, Path] = {}
    for result_file in result_files:
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        task_id = payload.get("task_id")
        if isinstance(task_id, str) and task_id:
            if expected_task_ids is None or task_id in expected_task_ids:
                latest_by_task[task_id] = result_file

    ordered_ids = (
        [tid for tid in expected_task_ids if tid in latest_by_task]
        if expected_task_ids is not None
        else sorted(latest_by_task)
    )

    entries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for task_id in ordered_ids:
        result_file = latest_by_task[task_id]
        run_dir = result_file.parent
        result_data: dict[str, Any] = json.loads(
            result_file.read_text(encoding="utf-8")
        )
        combined = _find_combined_score(run_dir)

        metrics_block = result_data.get("metrics")
        token_usage = (
            metrics_block.get("token_usage")
            if isinstance(metrics_block, dict)
            else None
        )
        input_tokens = (
            _coerce_int(token_usage.get("input_tokens"))
            if isinstance(token_usage, dict)
            else 0
        )
        output_tokens = (
            _coerce_int(token_usage.get("output_tokens"))
            if isinstance(token_usage, dict)
            else 0
        )
        cached_tokens = (
            _coerce_int(token_usage.get("cached_input_tokens"))
            if isinstance(token_usage, dict)
            else 0
        )
        token_cost_block = (
            metrics_block.get("token_cost") if isinstance(metrics_block, dict) else None
        )
        cost_usd = (
            _coerce_float(token_cost_block.get("total_cost"))
            if isinstance(token_cost_block, dict)
            else 0.0
        )

        if combined is None:
            err_msg = str(
                result_data.get("skip")
                or (result_data.get("evaluation") or {}).get("error_message")
                or f"Missing combined_score.json in {run_dir}"
            )
            errors.append(
                {
                    "instance_id": task_id,
                    "error": err_msg,
                    "test_result": {},
                    "instruction": str(result_data.get("input_query") or ""),
                    "history": [],
                    "metrics": {
                        "accumulated_cost": cost_usd,
                        "total_cost_usd": cost_usd,
                        "total_prompt_tokens": input_tokens,
                        "total_completion_tokens": output_tokens,
                    },
                }
            )
            continue

        agent_pct, agent_pass, agent_passed_n, agent_total_n = _layer_pct(
            combined.get("agent_score")
        )
        harness_pct, harness_pass, harness_passed_n, harness_total_n = _layer_pct(
            combined.get("harness_score")
        )
        clean_pass = bool(combined.get("pass", agent_pass and harness_pass))
        overall_score = float(combined.get("score", (agent_pct + harness_pct) / 2.0))
        attribution_obj = combined.get("attribution")
        attribution_tag = (
            str(attribution_obj.get("attribution") or "")
            if isinstance(attribution_obj, dict)
            else ""
        )

        entries.append(
            {
                "instance_id": task_id,
                "test_result": {
                    "passed": agent_pass,
                    "agent_pass": agent_pass,
                    "harness_pass": harness_pass,
                    "clean_pass": clean_pass,
                    "agent_score_pct": agent_pct,
                    "harness_score_pct": harness_pct,
                    "overall_score": round(overall_score, 4),
                    "agent_items": f"{agent_passed_n}/{agent_total_n}",
                    "harness_items": f"{harness_passed_n}/{harness_total_n}",
                    "attribution": attribution_tag,
                    "dimension": task_id[0].upper() if task_id else "",
                    "run_dir": str(run_dir),
                },
                "instruction": str(result_data.get("input_query") or ""),
                "error": None,
                "history": [],
                "metrics": {
                    "accumulated_cost": cost_usd,
                    "total_cost_usd": cost_usd,
                    "total_prompt_tokens": input_tokens,
                    "total_completion_tokens": output_tokens,
                    "cached_prompt_tokens": cached_tokens,
                },
            }
        )

    if expected_task_ids is not None:
        seen_ids = {e["instance_id"] for e in entries} | {
            e["instance_id"] for e in errors
        }
        for missing_id in expected_task_ids:
            if missing_id not in seen_ids:
                errors.append(
                    {
                        "instance_id": missing_id,
                        "error": "No TraceTrial-Bench result directory produced",
                        "test_result": {},
                        "instruction": "",
                        "history": [],
                        "metrics": {
                            "accumulated_cost": 0.0,
                            "total_cost_usd": 0.0,
                            "total_prompt_tokens": 0,
                            "total_completion_tokens": 0,
                        },
                    }
                )

    if not entries and not errors:
        raise RuntimeError(f"No TraceTrial-Bench run outputs found in {runs_dir}")

    eval_output_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_output_path.open("w", encoding="utf-8") as out_handle:
        for item in entries:
            out_handle.write(json.dumps(item) + "\n")
        for item in errors:
            out_handle.write(json.dumps(item) + "\n")

    logger.info(
        "Converted %d completed + %d failed TraceTrial-Bench runs -> %s",
        len(entries),
        len(errors),
        eval_output_path,
    )
    return len(entries), len(errors)


def resolve_vendor_sdk_dir(use_vendor_sdk: bool) -> Path | None:
    """Return ``vendor/software-agent-sdk`` when present and enabled."""
    if not use_vendor_sdk:
        return None
    if (VENDOR_SDK_DIR / "openhands-sdk").is_dir() and (
        VENDOR_SDK_DIR / "openhands-tools"
    ).is_dir():
        return VENDOR_SDK_DIR
    logger.warning(
        "vendor/software-agent-sdk is not initialized at %s; falling back to "
        "PyPI openhands package inside the container.",
        VENDOR_SDK_DIR,
    )
    return None


def get_parser() -> argparse.ArgumentParser:
    """Build argument parser for ``tracetrialbench-infer``."""
    parser = argparse.ArgumentParser(
        description="Run TraceTrial-Bench harness reliability evaluation on OpenHands",
    )
    parser.add_argument(
        "llm_config_path",
        help="Path to JSON LLM configuration file (e.g. .llm_config/eval.json)",
    )
    parser.add_argument(
        "--judge-llm-config",
        default=None,
        help="Optional separate LLM config JSON for TraceTrial-Bench rubric judging "
        "(defaults to llm_config_path)",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_outputs",
        help="Base directory for evaluation outputs (default: eval_outputs)",
    )
    parser.add_argument(
        "--dataset",
        default=str(INFER_DEFAULTS["dataset"]),
        help=f"Dataset identifier for directory naming (default: {DATASET_NAME})",
    )
    parser.add_argument(
        "--split",
        default=str(INFER_DEFAULTS["dataset_split"]),
        help="Dataset split label (default: final)",
    )
    parser.add_argument(
        "--manifest",
        default=str(INFER_DEFAULTS["manifest"]),
        help="Relative manifest path inside TraceTrial-Bench (default: Task/final.manifest)",
    )
    parser.add_argument(
        "--select",
        default=None,
        help="Optional text file listing specific TraceTrial-Bench task IDs (one per line)",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=0,
        help="Maximum number of tasks to run (0 = all tasks in manifest)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=int(INFER_DEFAULTS["max_iterations"]),
        help="Max iterations label for output directory naming (default: 100)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(INFER_DEFAULTS["num_workers"]),
        help="Concurrent Docker task workers (default: 4)",
    )
    parser.add_argument(
        "--note",
        default="default",
        help="Experiment note suffix for output directory naming",
    )
    parser.add_argument(
        "--use-vendor-sdk",
        action=argparse.BooleanOptionalAction,
        default=bool(INFER_DEFAULTS["use_vendor_sdk"]),
        help="Overlay checked-out vendor/software-agent-sdk into the OpenHands "
        "container image (default: True)",
    )
    parser.add_argument(
        "--tracetrial-commit",
        default=TRACETRIAL_PINNED_COMMIT,
        help=f"Pinned TraceTrial-Bench commit (default: {TRACETRIAL_PINNED_COMMIT[:12]})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``tracetrialbench-infer``."""
    args = get_parser().parse_args(argv)
    agent_llm = load_llm_config(args.llm_config_path)
    judge_llm = (
        load_llm_config(args.judge_llm_config) if args.judge_llm_config else None
    )

    structured_output_dir = Path(
        construct_eval_output_dir(
            base_dir=args.output_dir,
            dataset_name=f"{args.dataset}-{args.split}",
            model_name=agent_llm.model,
            max_iterations=args.max_iterations,
            eval_note=args.note,
        )
    )
    structured_output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = structured_output_dir / OUTPUT_FILENAME
    runs_dir = structured_output_dir / "tracetrial_runs"
    image_builds_dir = structured_output_dir / "image_builds"

    cli_bin = ensure_tracetrial_checkout(commit=args.tracetrial_commit)
    manifest_path = REPO_CACHE_DIR / args.manifest
    all_task_ids = load_manifest_task_ids(manifest_path)
    selected_ids = select_task_ids(
        all_task_ids,
        selected_file=args.select,
        n_limit=args.n_limit,
    )
    logger.info(
        "Selected %d/%d TraceTrial-Bench task(s) from %s",
        len(selected_ids),
        len(all_task_ids),
        args.manifest,
    )

    env = dict(os.environ)
    sdk_dir = resolve_vendor_sdk_dir(args.use_vendor_sdk)
    if sdk_dir is not None:
        env["HARNESSBENCH_OPENHANDS_SDK_DIR"] = str(sdk_dir)
        logger.info("Overlaying vendor/software-agent-sdk from %s", sdk_dir)

    with tempfile.TemporaryDirectory(prefix="tracetrial-cfg-") as tmp_cfg:
        tmp_cfg_dir = Path(tmp_cfg)
        model_toml_path = write_private_model_toml(
            build_tracetrial_model_toml(agent_llm, judge_llm),
            tmp_cfg_dir / "model.toml",
        )
        subset_manifest = tmp_cfg_dir / "selected.manifest"
        subset_manifest.write_text(
            "\n".join(f"Task/{tid}" for tid in selected_ids) + "\n",
            encoding="utf-8",
        )

        logger.info("Building TraceTrial-Bench OpenHands container image...")
        _run_checked(
            [
                str(cli_bin),
                "image",
                "build",
                "--harness",
                "openhands",
                "--model-registry",
                str(model_toml_path),
                "--output-dir",
                str(image_builds_dir),
            ],
            cwd=REPO_CACHE_DIR,
            env=env,
            error_prefix="TraceTrial-Bench OpenHands image build failed",
        )

        judge_alias = "eval_judge" if judge_llm is not None else "eval_agent"
        run_cmd = [
            str(cli_bin),
            "run",
            "--task-manifest",
            str(subset_manifest),
            "--harness",
            "openhands",
            "--model",
            "eval_agent",
            "--judge-model",
            judge_alias,
            "--mode",
            "docker",
            "--concurrency",
            str(max(1, args.num_workers)),
            "--model-registry",
            str(model_toml_path),
            "--output-dir",
            str(runs_dir),
        ]
        logger.info(
            "Running TraceTrial-Bench tasks (%d instances)...", len(selected_ids)
        )
        completed = subprocess.run(
            run_cmd,
            cwd=str(REPO_CACHE_DIR),
            env=env,
        )
        if completed.returncode != 0:
            logger.warning(
                "tracetrial_bench run exited with status %d; collecting produced "
                "task scorecards.",
                completed.returncode,
            )

    convert_tracetrial_runs_to_eval_output(
        runs_dir,
        output_jsonl,
        expected_task_ids=selected_ids,
    )
    generate_cost_report(str(output_jsonl))
    logger.info("TraceTrial-Bench inference complete: %s", output_jsonl)


if __name__ == "__main__":
    main()
