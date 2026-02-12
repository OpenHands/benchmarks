from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.scripts.generate_llm_config import generate_config


INFER_ENTRYPOINTS = {
    "swebench": "swebench-infer",
    "gaia": "gaia-infer",
    "commit0": "commit0-infer",
    "multiswebench": "multiswebench-infer",
    "swtbench": "swtbench-infer",
    "swebenchmultimodal": "swebenchmultimodal-infer",
    "openagentsafety": "openagentsafety-infer",
}

EVAL_ENTRYPOINTS = {
    "swebench": "swebench-eval",
    "gaia": "gaia-eval",
    "commit0": "commit0-eval",
    "multiswebench": "multiswebench-eval",
    "swtbench": "swtbench-eval",
    "swebenchmultimodal": "swebenchmultimodal-eval",
    # openagentsafety doesn't have a separate eval entrypoint
}

# Patch-based benchmarks use "finish_with_patch" (requires git patch).
# gaia and openagentsafety use "pass" (accept any completed output).
BENCHMARK_CRITIC = {
    "swebench": "finish_with_patch",
    "swtbench": "finish_with_patch",
    "swebenchmultimodal": "finish_with_patch",
    "multiswebench": "finish_with_patch",
    "commit0": "finish_with_patch",
    "gaia": "pass",
    "openagentsafety": "pass",
}


def _build_infer_cmd(args: argparse.Namespace, llm_config_path: Path) -> list[str]:
    """Build the inference command with benchmark-specific args."""
    cmd = [
        INFER_ENTRYPOINTS[args.benchmark],
        str(llm_config_path),
        "--workspace", args.workspace,
        "--max-iterations", str(args.max_iterations),
        "--conversation-timeout", str(args.conversation_timeout),
        "--num-workers", str(args.num_workers),
        "--output-dir", str(args.output_dir),
        "--max-attempts", str(args.max_attempts),
        "--max-retries", str(args.instance_max_retries),
        "--critic", BENCHMARK_CRITIC.get(args.benchmark, "finish_with_patch"),
    ]
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    if args.split:
        cmd.extend(["--split", args.split])

    if args.note:
        cmd.extend(["--note", args.note])
    if args.n_limit is not None:
        cmd.extend(["--n-limit", str(args.n_limit)])
    if args.skip_failed_samples:
        cmd.append("--skip-failed-samples")

    # ----- Benchmark-specific inference args -----

    # GAIA requires --level (e.g. 2023_level1, 2023_all)
    if args.benchmark == "gaia" and args.level:
        cmd.extend(["--level", args.level])

    # commit0 requires --repo-split (e.g. lite, all)
    if args.benchmark == "commit0" and args.repo_split:
        cmd.extend(["--repo-split", args.repo_split])

    # multiswebench requires --lang (e.g. java, python, go, c)
    if args.benchmark == "multiswebench" and args.language:
        cmd.extend(["--lang", args.language])

    return cmd


def _build_eval_cmd(args: argparse.Namespace, output_jsonl: Path) -> list[str]:
    """Build the evaluation command with benchmark-specific args."""
    benchmark = args.benchmark
    if benchmark not in EVAL_ENTRYPOINTS:
        return []

    cmd = [EVAL_ENTRYPOINTS[benchmark], str(output_jsonl)]

    if benchmark in ("swebench", "swebenchmultimodal") and args.dataset:
        cmd.extend(["--dataset", args.dataset])

    if benchmark == "swebench":
        cmd.extend(["--run-id", "eval"])
    if benchmark in ("swebench", "swebenchmultimodal"):
        if args.modal is True:
            cmd.append("--modal")
        elif args.modal is False:
            cmd.append("--no-modal")

    if benchmark == "multiswebench" and args.dataset:
        cmd.extend(["--dataset", args.dataset])
        if args.language:
            cmd.extend(["--lang", args.language])

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()

    # LLM config generation args
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--api-base-url", type=str, required=True)
    parser.add_argument("--api-key-env", type=str, default=None, help="Env var name for API key")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=3)

    # Benchmark selection
    parser.add_argument("--benchmark", required=True, choices=INFER_ENTRYPOINTS.keys())

    # Common inference args
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--workspace", type=str, default="docker")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--conversation-timeout", type=float, default=3600.0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--instance-max-retries", type=int, default=3)
    parser.add_argument("--n-limit", type=int, default=None)
    parser.add_argument("--skip-failed-samples", action="store_true")

    # GAIA
    parser.add_argument("--level", type=str, default="2023_all",
                        help="GAIA level (e.g. 2023_level1, 2023_all)")
    # commit0
    parser.add_argument("--repo-split", type=str, default="lite",
                        help="commit0 repo split (lite, all, or repo name)")
    # multiswebench
    parser.add_argument("--language", type=str, default=None,
                        help="multiswebench language (java, python, go, c)")
    # swebench/swebenchmultimodal
    parser.add_argument(
        "--modal",
        dest="modal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable Modal for swebench and swebenchmultimodal evaluation. "
            "If omitted, each benchmark uses its default."
        ),
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_config_path = output_dir / "llm_config.json"

    # 1) Generate LLM config
    generate_config(
        model=args.model,
        api_base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        output_path=str(llm_config_path),
    )

    # 2) Run inference
    # multiswebench reads LANGUAGE env var at module level for Docker image naming
    if args.benchmark == "multiswebench" and args.language:
        os.environ["LANGUAGE"] = args.language

    infer_cmd = _build_infer_cmd(args, llm_config_path)
    ret = subprocess.call(infer_cmd)
    if ret != 0:
        sys.exit(ret)

    # 3) Find output.jsonl and run evaluation
    output_files = sorted(output_dir.rglob("output.jsonl"))
    if not output_files:
        print(f"ERROR: Inference did not produce output.jsonl under {output_dir}", file=sys.stderr)
        sys.exit(1)

    output_jsonl = output_files[-1]  # Use the latest one

    eval_cmd = _build_eval_cmd(args, output_jsonl)
    if eval_cmd:
        sys.exit(subprocess.call(eval_cmd))


if __name__ == "__main__":
    main()
