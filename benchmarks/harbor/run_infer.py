#!/usr/bin/env python3
"""Run any Harbor dataset/config/path with the OpenHands SDK agent."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.utils.evaluation_utils import construct_eval_output_dir
from benchmarks.utils.harbor import (
    _secret_value,
    check_harbor_installed,
    convert_harbor_to_eval_output,
)
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import LLM, get_logger


logger = get_logger(__name__)
OUTPUT_FILENAME = "output.jsonl"
DEFAULT_ADAPTER_REPO = "https://github.com/harbor-framework/harbor.git"


def _load_task_ids(filepath: str) -> list[str]:
    task_ids: list[str] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value and not value.startswith("#"):
                task_ids.append(value)
    return task_ids


def _checkout_adapter(repo: str, ref: str | None) -> tuple[Path, str]:
    """Clone the adapter repo and return (checkout_dir, resolved_commit_sha).

    The resolved SHA is captured so metadata can record exactly which commit
    was evaluated, making runs reproducible even when ``ref`` is unset.
    """
    if not ref:
        logger.warning(
            "Cloning adapter repo without a pinned ref; results may not be "
            "reproducible. Pass --harbor-adapter-ref to pin a tag/SHA/branch."
        )
    checkout_dir = Path(tempfile.mkdtemp(prefix="harbor-adapter-"))
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd.extend(["--branch", ref])
    cmd.extend([repo, str(checkout_dir)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and ref:
        logger.warning("Shallow clone by ref failed; retrying full fetch checkout")
        shutil.rmtree(checkout_dir, ignore_errors=True)
        checkout_dir = Path(tempfile.mkdtemp(prefix="harbor-adapter-"))
        result = subprocess.run(
            ["git", "clone", repo, str(checkout_dir)], capture_output=True, text=True
        )
        if result.returncode == 0:
            result = subprocess.run(
                ["git", "checkout", ref],
                cwd=checkout_dir,
                capture_output=True,
                text=True,
            )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to checkout Harbor adapter repo: {result.stderr}")
    sha_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout_dir, capture_output=True, text=True
    )
    resolved_sha = (
        sha_result.stdout.strip() if sha_result.returncode == 0 else "unknown"
    )
    return checkout_dir, resolved_sha


def _resolve_target(
    args: argparse.Namespace,
) -> tuple[str, str, str | None, str | None]:
    checkout_dir: Path | None = None
    adapter_sha: str | None = None
    target = args.harbor_target
    target_type = args.harbor_target_type

    if args.harbor_adapter_repo or args.harbor_adapter_path:
        repo = args.harbor_adapter_repo or DEFAULT_ADAPTER_REPO
        checkout_dir, adapter_sha = _checkout_adapter(repo, args.harbor_adapter_ref)
        if args.harbor_adapter_path:
            target_path = checkout_dir / args.harbor_adapter_path
            if not target_path.exists():
                raise RuntimeError(f"Harbor adapter path does not exist: {target_path}")
            target = str(target_path)
            if target_type == "auto":
                target_type = (
                    "config" if target_path.suffix in {".yaml", ".yml"} else "path"
                )

    if not target:
        raise RuntimeError("A Harbor target or adapter path is required")

    if target_type == "auto":
        path = Path(target)
        if path.exists():
            target_type = "config" if path.suffix in {".yaml", ".yml"} else "path"
        else:
            target_type = "dataset"

    return (
        target,
        target_type,
        str(checkout_dir) if checkout_dir else None,
        adapter_sha,
    )


SECRET_KEY_PATTERNS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PASSPHRASE")
SENSITIVE_VALUE_FLAGS = ("--ae", "--ak")


def _is_sensitive_value(prev: str, part: str) -> bool:
    """Return True if ``part`` is a value following ``--ae``/``--ak`` whose key looks secret."""
    if prev not in SENSITIVE_VALUE_FLAGS:
        return False
    key = part.split("=", 1)[0].upper()
    return any(pat in key for pat in SECRET_KEY_PATTERNS)


def _target_args(target: str, target_type: str) -> list[str]:
    if target_type == "dataset":
        return ["-d", target]
    if target_type == "config":
        return ["-c", target]
    if target_type == "path":
        return ["-p", target]
    raise ValueError(f"Unsupported Harbor target type: {target_type}")


def _parse_key_value(values: list[str]) -> list[str]:
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE, got {value!r}")
    return values


def _split_json_values(raw: str | None) -> list[str]:
    if not raw:
        return []
    data = json.loads(raw)
    if isinstance(data, dict):
        return [f"{key}={value}" for key, value in data.items()]
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        return data
    raise ValueError("Expected a JSON object or list of KEY=VALUE strings")


def run_harbor(
    args: argparse.Namespace,
    llm: LLM,
    output_dir: str,
    target: str,
    target_type: str,
    checkout_dir: str | None = None,
    adapter_sha: str | None = None,
) -> Path:
    harbor_output_dir = Path(output_dir) / "harbor_output"
    harbor_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.harbor_executable,
        "run",
        *_target_args(target, target_type),
        "-a",
        args.harbor_agent,
        "-m",
        llm.model,
        "--jobs-dir",
        str(harbor_output_dir.resolve()),
        "--n-concurrent",
        str(args.num_workers),
    ]

    if llm.api_key:
        cmd.extend(["--ae", f"LLM_API_KEY={_secret_value(llm.api_key)}"])
    if llm.base_url:
        cmd.extend(["--ae", f"LLM_BASE_URL={llm.base_url}"])
    for env_value in _parse_key_value(
        [*args.agent_env, *_split_json_values(args.agent_env_json)]
    ):
        cmd.extend(["--ae", env_value])
    for kwarg_value in _parse_key_value(
        [*args.agent_kwarg, *_split_json_values(args.agent_kwarg_json)]
    ):
        cmd.extend(["--ak", kwarg_value])
    for task_id in args.task_id or []:
        task_value = task_id.rsplit("/", 1)[-1] if target_type == "path" else task_id
        cmd.extend([args.task_filter_flag, task_value])
    if args.n_limit is not None:
        cmd.extend(["--n-tasks", str(args.n_limit)])
    for extra_arg in args.harbor_arg:
        cmd.extend(shlex.split(extra_arg))

    safe_cmd = [
        "***" if _is_sensitive_value(prev, part) else part
        for prev, part in zip([""] + cmd, cmd)
    ]
    logger.info("Running Harbor command: %s", " ".join(safe_cmd))
    if checkout_dir:
        logger.info("Using Harbor adapter checkout: %s", checkout_dir)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Harbor stdout: %s", result.stdout)
        logger.error("Harbor stderr: %s", result.stderr)
        raise RuntimeError(
            f"Harbor run failed with exit code {result.returncode}: {result.stderr}"
        )
    logger.info("Harbor stdout: %s", result.stdout)
    return harbor_output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a generic Harbor evaluation with OpenHands SDK"
    )
    parser.add_argument("llm_config_path")
    parser.add_argument(
        "--harbor-target", help="Harbor dataset name, config path, or dataset path"
    )
    parser.add_argument(
        "--harbor-target-type",
        choices=["auto", "dataset", "config", "path"],
        default="auto",
    )
    parser.add_argument(
        "--harbor-adapter-repo", help="Git repository containing the Harbor adapter"
    )
    parser.add_argument("--harbor-adapter-ref", help="Git ref/SHA/tag for adapter repo")
    parser.add_argument(
        "--harbor-adapter-path",
        help="Path inside adapter repo to a Harbor YAML config or dataset directory",
    )
    parser.add_argument("--harbor-agent", default="openhands-sdk")
    parser.add_argument("--harbor-executable", default="harbor")
    parser.add_argument("--task-filter-flag", default="--include-task-name")
    parser.add_argument(
        "--agent-env", action="append", default=[], help="KEY=VALUE passed as --ae"
    )
    parser.add_argument("--agent-env-json", help="JSON object passed as repeated --ae")
    parser.add_argument(
        "--agent-kwarg", action="append", default=[], help="KEY=VALUE passed as --ak"
    )
    parser.add_argument(
        "--agent-kwarg-json", help="JSON object passed as repeated --ak"
    )
    parser.add_argument(
        "--harbor-arg", action="append", default=[], help="Additional raw Harbor args"
    )
    parser.add_argument("--benchmark-slug", default="harbor")
    parser.add_argument("--output-dir", default="./evaluation_outputs")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--n-limit", type=int)
    parser.add_argument("--select", help="Text file containing task IDs")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--note")
    parser.add_argument("--skip-harbor", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()

    if not os.path.isfile(args.llm_config_path):
        logger.error("LLM config file does not exist: %s", args.llm_config_path)
        sys.exit(1)
    with open(args.llm_config_path, encoding="utf-8") as f:
        llm = LLM.model_validate_json(f.read())

    if args.select:
        args.task_id = [*(args.task_id or []), *_load_task_ids(args.select)]

    if not args.skip_harbor and not check_harbor_installed(args.harbor_executable):
        logger.error("Harbor CLI is not installed; install with `pip install harbor`.")
        sys.exit(1)

    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=args.benchmark_slug,
        model_name=llm.model,
        # Standard iteration cap used by all benchmark runners in this repo
        max_iterations=100,
        eval_note=args.note,
    )
    os.makedirs(structured_output_dir, exist_ok=True)

    target, target_type, checkout_dir, adapter_sha = _resolve_target(args)
    metadata = {
        "llm": llm.model_dump_json(),
        "benchmark": args.benchmark_slug,
        "harbor_target": target,
        "harbor_target_type": target_type,
        "harbor_adapter_repo": args.harbor_adapter_repo,
        "harbor_adapter_ref": args.harbor_adapter_ref,
        "harbor_adapter_resolved_sha": adapter_sha,
        "harbor_adapter_path": args.harbor_adapter_path,
        "harbor_adapter_checkout": checkout_dir,
        "harbor_agent": args.harbor_agent,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": args.note,
    }
    with open(
        Path(structured_output_dir) / "metadata.json", "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    output_path = Path(structured_output_dir) / OUTPUT_FILENAME
    try:
        harbor_output_dir = (
            Path(structured_output_dir) / "harbor_output"
            if args.skip_harbor
            else run_harbor(
                args,
                llm,
                structured_output_dir,
                target,
                target_type,
                checkout_dir,
                adapter_sha,
            )
        )
        convert_harbor_to_eval_output(
            harbor_output_dir=harbor_output_dir, eval_output_path=output_path
        )
    except Exception as exc:
        logger.error("Harbor inference failed: %s", exc)
        sys.exit(1)
    finally:
        if checkout_dir:
            shutil.rmtree(checkout_dir, ignore_errors=True)

    if output_path.exists():
        generate_cost_report(str(output_path))
    print(json.dumps({"output_json": str(output_path)}))


if __name__ == "__main__":
    main()
