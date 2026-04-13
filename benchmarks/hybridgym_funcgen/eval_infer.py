#!/usr/bin/env python3
"""
Hybrid-Gym func_gen evaluation script.

Reads output.jsonl from run_infer.py and evaluates generated functions
using RepoST's eval_script tests. Each instance carries an eval_script
that compares the agent's implementation against the reference.

Usage:
    uv run hybridgym-funcgen-eval output.jsonl --run-id my_run
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from benchmarks.utils.laminar import LaminarService
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)

DOCKER_IMAGE = "yiqingxyq/repost:v0"


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------


@contextmanager
def docker_container(image: str = DOCKER_IMAGE):
    """Start a persistent Docker container; clean up on exit."""
    container_id = None
    try:
        name = f"repost_eval_{uuid.uuid4().hex[:8]}"
        result = subprocess.run(
            ["docker", "run", "-d", "--name", name, image, "tail", "-f", "/dev/null"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")
        container_id = result.stdout.strip()
        logger.info("Started container %s", name)
        yield container_id
    finally:
        if container_id:
            subprocess.run(
                ["docker", "stop", container_id], capture_output=True, timeout=30
            )
            subprocess.run(
                ["docker", "rm", container_id], capture_output=True, timeout=30
            )


def run_test_in_container(
    test_content: str, container_id: str, timeout: int = 60
) -> dict:
    test_file = f"/tmp/test_{uuid.uuid4().hex[:8]}.py"
    try:
        write = subprocess.run(
            [
                "docker",
                "exec",
                container_id,
                "bash",
                "-c",
                f"cat > {test_file} << 'EOFTEST'\n{test_content}\nEOFTEST",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if write.returncode != 0:
            return {"success": False, "error": f"write failed: {write.stderr}"}

        result = subprocess.run(
            ["docker", "exec", container_id, "python", test_file],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout"}
    finally:
        subprocess.run(
            ["docker", "exec", container_id, "rm", "-f", test_file],
            capture_output=True,
            timeout=10,
        )


# ---------------------------------------------------------------------------
# Test construction
# ---------------------------------------------------------------------------


def rename_function(code: str, old_name: str, new_name: str) -> str:
    return re.sub(rf"((?:async\s+)?def\s+){old_name}(\s*\()", rf"\1{new_name}\2", code)


def construct_test_file(
    generated_function: str, eval_script: str, func_name: str
) -> str:
    if "." in func_name:
        return _construct_for_method(generated_function, eval_script, func_name)
    return _construct_for_function(generated_function, eval_script, func_name)


def _construct_for_function(gen: str, script: str, func_name: str) -> str:
    new_name = f"{func_name}_new_implementation"
    renamed = rename_function(gen, func_name, new_name)
    lines = script.split("\n")
    imports, rest = [], []
    in_imports = True
    for line in lines:
        s = line.strip()
        if in_imports and (
            s.startswith("import ")
            or s.startswith("from ")
            or s.startswith("#")
            or s == ""
        ):
            imports.append(line)
        else:
            in_imports = False
            rest.append(line)
    return "\n".join(imports) + f"\n\n{renamed}\n\n" + "\n".join(rest)


def _construct_for_method(gen: str, script: str, func_name: str) -> str:
    cls_name, method_name = func_name.rsplit(".", 1)
    new_name = f"{method_name}_new_implementation"
    renamed = re.sub(
        rf"((?:async\s+)?def\s+){method_name}(\s*\()", rf"\1{new_name}\2", gen
    )

    # Ensure 4-space indent for class methods
    first = renamed.split("\n")[0]
    cur_indent = len(first) - len(first.lstrip())
    if cur_indent != 4:
        adjusted = []
        for line in renamed.split("\n"):
            if line.strip():
                rel = (len(line) - len(line.lstrip())) - cur_indent
                adjusted.append(" " * (4 + rel) + line.lstrip())
            else:
                adjusted.append(line)
        renamed = "\n".join(adjusted)

    # Find insertion point in eval_script
    pattern = rf"((?:    @\w+.*\n)*)    def {method_name}\s*\("
    match = re.search(pattern, script)
    if match:
        decorators = match.group(1)
        if decorators.strip():
            renamed = decorators + renamed
        pos = match.start()
    else:
        fallback = re.search(rf"    def {method_name}\s*\(", script)
        if fallback:
            pos = fallback.start()
        else:
            return _construct_for_function(gen, script, func_name)

    merged = script[:pos] + renamed + "\n\n" + script[pos:]
    lines = merged.split("\n")
    imports, rest = [], []
    in_imports = True
    for line in lines:
        s = line.strip()
        if in_imports and (
            s.startswith("import ")
            or s.startswith("from ")
            or s.startswith("#")
            or s == ""
        ):
            imports.append(line)
        else:
            in_imports = False
            rest.append(line)
    return "\n".join(imports) + "\n\n" + "\n".join(rest)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_instance(
    entry: dict, container_id: str | None = None, timeout: int = 60
) -> dict:
    instance_id = entry.get("instance_id", "unknown")
    test_result = entry.get("test_result", {})
    instance = entry.get("instance", {})

    gen_func = test_result.get("generated_function", "")
    eval_script = instance.get("eval_script", "")
    func_name = instance.get("func_name", "")

    if not gen_func or not eval_script or not func_name:
        reason = (
            "no_generated_function"
            if not gen_func
            else ("no_eval_script" if not eval_script else "no_func_name")
        )
        return {"instance_id": instance_id, "success": False, "reason": reason}

    test_content = construct_test_file(gen_func, eval_script, func_name)

    if container_id:
        result = run_test_in_container(test_content, container_id, timeout)
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_content)
            tmp = f.name
        try:
            r = subprocess.run(
                [sys.executable, tmp], capture_output=True, text=True, timeout=timeout
            )
            result = {
                "success": r.returncode == 0,
                "output": r.stdout,
                "error": r.stderr,
            }
        except subprocess.TimeoutExpired:
            result = {"success": False, "error": "timeout"}
        finally:
            Path(tmp).unlink(missing_ok=True)

    return {
        "instance_id": instance_id,
        "success": result["success"],
        "reason": "passed" if result["success"] else "test_failed",
    }


def generate_report(input_file: str, output_file: str, use_docker: bool) -> None:
    entries = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    results: list[dict] = []
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    error_ids: list[str] = []

    def run_evals(cid=None):
        for entry in entries:
            iid = entry.get("instance_id", "")
            if entry.get("error"):
                error_ids.append(iid)
                continue
            r = evaluate_instance(entry, container_id=cid)
            results.append(r)
            if r["success"]:
                resolved_ids.append(iid)
            else:
                unresolved_ids.append(iid)

    if use_docker:
        with docker_container() as cid:
            run_evals(cid)
    else:
        run_evals()

    submitted_ids = resolved_ids + unresolved_ids + error_ids
    report = {
        "schema_version": 2,
        "total_instances": len(submitted_ids),
        "submitted_instances": len(submitted_ids),
        "resolved_instances": len(resolved_ids),
        "resolved_ids": resolved_ids,
        "unresolved_instances": len(unresolved_ids),
        "unresolved_ids": unresolved_ids,
        "error_instances": len(error_ids),
        "error_ids": error_ids,
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    total = len(resolved_ids) + len(unresolved_ids)
    logger.info("=== Evaluation Results ===")
    logger.info("Total evaluated: %d", total)
    logger.info(
        "Passed: %d/%d (%.1f%%)",
        len(resolved_ids),
        total,
        100 * len(resolved_ids) / max(total, 1),
    )
    logger.info("Errors: %d", len(error_ids))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid-Gym func_gen outputs",
    )
    parser.add_argument("input_file", help="Path to output.jsonl from inference")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument(
        "--no-docker", action="store_true", help="Run tests without Docker"
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        sys.exit(1)

    report_path = input_path.with_suffix(".report.json")

    try:
        generate_report(
            str(input_path), str(report_path), use_docker=not args.no_docker
        )
        LaminarService.get().update_evaluation_scores(str(input_path), str(report_path))
        generate_cost_report(str(input_path))
        logger.info("Report saved to: %s", report_path)
        print(json.dumps({"report_json": str(report_path)}))
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
