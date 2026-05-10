"""ProgramBench evaluation script.

This wraps the upstream ``programbench eval`` CLI so we can plug ProgramBench
into the same reporting layout as our other benchmarks.

Workflow:
  1. ``run_infer.py`` writes ``<eval_output_dir>/run/<instance_id>/submission.tar.gz``.
  2. ``programbench eval <eval_output_dir>/run`` builds + runs the per-instance
     evaluation containers and writes ``<instance_id>.eval.json`` next to each
     submission.
  3. We then aggregate those JSON files into the standard report.json that
     downstream tooling (push-to-index, dashboards, ...) consumes.

Usage:
    uv run programbench-eval ./eval_outputs/.../output.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.programbench.config import EVAL_DEFAULTS
from benchmarks.programbench.run_infer import RUN_SUBDIR
from benchmarks.utils.laminar import LaminarService
from benchmarks.utils.report_costs import generate_cost_report
from openhands.sdk import get_logger


logger = get_logger(__name__)


def _resolve_run_dir(output_jsonl: Path) -> Path:
    """Locate the ``run/`` directory containing per-instance submissions.

    ``run_infer.py`` lays it out as a sibling of ``output.jsonl`` inside the
    evaluation output directory.
    """
    eval_dir = output_jsonl.parent
    run_dir = eval_dir / RUN_SUBDIR
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Expected ProgramBench submissions under {run_dir}; did "
            f"run_infer.py finish successfully?"
        )
    return run_dir


def _read_submitted_ids(output_jsonl: Path) -> list[str]:
    """Read instance ids from ``output.jsonl`` (one per line, JSON object)."""
    ids: list[str] = []
    if not output_jsonl.exists():
        logger.warning("output.jsonl missing at %s; treating as empty", output_jsonl)
        return ids
    with open(output_jsonl) as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error("output.jsonl line %d invalid JSON: %s", line_num, exc)
                continue
            iid = payload.get("instance_id")
            if iid:
                ids.append(iid)
    return ids


def _run_programbench_eval(
    run_dir: Path,
    *,
    workers: int,
    branch_workers: int,
    docker_cpus: int,
    image_tag: str,
    force: bool,
    timeout: float | None,
    extra_args: list[str] | None = None,
) -> int:
    """Invoke ``programbench eval <run_dir>`` and return its exit code.

    ``timeout`` is a wall-clock cap (seconds). ``programbench eval`` itself
    has no global timeout knob and a single hung docker container — e.g.
    a misbehaving submission whose entrypoint blocks forever — would
    otherwise leave the eval pod alive indefinitely. When the timeout
    fires we kill the subprocess and surface a non-zero exit code so the
    caller can still aggregate whatever per-instance JSON landed before
    the hang.
    """
    cli = shutil.which("programbench")
    if cli is None:
        raise RuntimeError(
            "The 'programbench' CLI is not on PATH. Install the upstream "
            "package first (`uv pip install programbench`)."
        )
    cmd = [
        cli,
        "eval",
        str(run_dir),
        "--workers",
        str(workers),
        "--branch-workers",
        str(branch_workers),
        "--docker-cpus",
        str(docker_cpus),
        "--image-tag",
        image_tag,
    ]
    if force:
        cmd.append("--force")
    if extra_args:
        cmd.extend(extra_args)
    logger.info("Running %s", " ".join(cmd))
    try:
        completed = subprocess.run(cmd, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error(
            "programbench eval exceeded timeout of %ss; killing subprocess "
            "and aggregating partial results.",
            timeout,
        )
        return 124  # GNU `timeout` convention: exit 124 on timeout
    return completed.returncode


def aggregate_eval_results(
    run_dir: Path,
    submitted_ids: list[str],
) -> dict[str, Any]:
    """Aggregate ``<id>/<id>.eval.json`` files into a benchmark report.

    The upstream eval JSON (see ProgramBench docs) contains, per instance:
      - ``test_results``: list of ``{name, branch, status, extra}`` records.
      - ``error_code``/``error_details``: top-level errors (null on success).
      - ``test_branches``/``test_branch_errors``: per-test-branch outcomes.

    We treat an instance as **resolved** iff:
      * ``error_code`` is null **and**
      * every entry in ``test_results`` has ``status == "passed"``.

    "Almost resolved" (>=95% pass-rate, used by the upstream leaderboard) is
    surfaced as a separate count so users can compare against it.
    """
    completed_ids: set[str] = set()
    resolved_ids: set[str] = set()
    almost_resolved_ids: set[str] = set()
    unresolved_ids: set[str] = set()
    error_ids: set[str] = set()

    for iid in sorted(set(submitted_ids)):
        eval_path = run_dir / iid / f"{iid}.eval.json"
        if not eval_path.exists():
            error_ids.add(iid)
            continue
        try:
            payload = json.loads(eval_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read %s: %s", eval_path, exc)
            error_ids.add(iid)
            continue

        completed_ids.add(iid)
        if payload.get("error_code"):
            error_ids.add(iid)
            unresolved_ids.add(iid)
            continue

        results = payload.get("test_results") or []
        if not results:
            # Empty test_results with no error_code means the eval ran but
            # no behavioral tests executed — count as unresolved, not error,
            # because the harness completed.
            unresolved_ids.add(iid)
            continue

        passed = sum(1 for r in results if r.get("status") == "passed")
        total = len(results)
        ratio = passed / total
        if ratio == 1.0:
            resolved_ids.add(iid)
        else:
            unresolved_ids.add(iid)
            if ratio >= 0.95:
                almost_resolved_ids.add(iid)

    submitted_set = set(submitted_ids)
    incomplete_ids = submitted_set - completed_ids

    return {
        "total_instances": len(submitted_set),
        "submitted_instances": len(submitted_set),
        "completed_instances": len(completed_ids),
        "incomplete_instances": len(incomplete_ids),
        "resolved_instances": len(resolved_ids),
        "almost_resolved_instances": len(almost_resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "error_instances": len(error_ids),
        "submitted_ids": sorted(submitted_set),
        "completed_ids": sorted(completed_ids),
        "incomplete_ids": sorted(incomplete_ids),
        "resolved_ids": sorted(resolved_ids),
        "almost_resolved_ids": sorted(almost_resolved_ids),
        "unresolved_ids": sorted(unresolved_ids),
        "error_ids": sorted(error_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run programbench eval over a benchmarks output directory "
        "and aggregate per-instance JSON into a benchmarks-style report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run programbench-eval ./eval_outputs/.../output.jsonl
    uv run programbench-eval ./eval_outputs/.../output.jsonl --skip-eval
    uv run programbench-eval ./eval_outputs/.../output.jsonl --workers 4
        """,
    )
    parser.add_argument("input_file", help="Path to the run_infer output.jsonl file")
    parser.add_argument(
        "--output-file",
        help="Where to write the aggregated report (default: <input>.report.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=EVAL_DEFAULTS["workers"],
        help="programbench eval --workers value (default: 1)",
    )
    parser.add_argument(
        "--branch-workers",
        type=int,
        default=EVAL_DEFAULTS["branch_workers"],
        help="programbench eval --branch-workers value (default: 1)",
    )
    parser.add_argument(
        "--docker-cpus",
        type=int,
        default=EVAL_DEFAULTS["docker_cpus"],
        help="programbench eval --docker-cpus value (default: 10)",
    )
    parser.add_argument(
        "--image-tag",
        default=EVAL_DEFAULTS["image_tag"],
        help="Docker image tag to evaluate (default: task)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to programbench eval (re-eval already-graded instances).",
    )
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=float(os.environ.get("PROGRAMBENCH_EVAL_TIMEOUT", "0") or 0) or None,
        help=(
            "Wall-clock timeout (seconds) for the underlying `programbench "
            "eval` subprocess. 0 or unset disables. Also configurable via "
            "the PROGRAMBENCH_EVAL_TIMEOUT environment variable. Useful in "
            "CI to bound a hung docker container; partial per-instance "
            "JSON is still aggregated on timeout."
        ),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip running programbench eval and only aggregate existing "
        "<id>/<id>.eval.json files.",
    )
    args = parser.parse_args()

    input_file = Path(args.input_file).resolve()
    if not input_file.exists():
        logger.error("Input file does not exist: %s", input_file)
        sys.exit(1)

    output_file = (
        Path(args.output_file)
        if args.output_file
        else input_file.with_suffix(".report.json")
    )
    run_dir = _resolve_run_dir(input_file)

    if not args.skip_eval:
        rc = _run_programbench_eval(
            run_dir,
            workers=args.workers,
            branch_workers=args.branch_workers,
            docker_cpus=args.docker_cpus,
            image_tag=args.image_tag,
            force=args.force,
            timeout=args.eval_timeout,
        )
        if rc != 0:
            # Don't exit here: programbench eval can return non-zero when
            # individual instances fail, but partial JSON output is still
            # useful for the report. Surface the failure so the user sees it
            # and can decide to investigate.
            logger.warning(
                "programbench eval exited with rc=%d; continuing with "
                "aggregation of whatever JSON was produced.",
                rc,
            )

    submitted_ids = _read_submitted_ids(input_file)
    report = aggregate_eval_results(run_dir, submitted_ids)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as fh:
        json.dump(report, fh, indent=4)
    logger.info("Wrote report: %s", output_file)
    logger.info(
        "Resolved %d / %d instances (almost: %d, errors: %d)",
        report["resolved_instances"],
        report["total_instances"],
        report["almost_resolved_instances"],
        report["error_instances"],
    )

    # Best-effort telemetry/cost reporting; never fail the eval over these.
    try:
        LaminarService.get().update_evaluation_scores(str(input_file), str(output_file))
    except Exception as exc:  # pragma: no cover - telemetry only
        logger.warning("Laminar update failed (non-critical): %s", exc)
    try:
        generate_cost_report(str(input_file))
    except Exception as exc:  # pragma: no cover - reporting only
        logger.warning("Cost report generation failed (non-critical): %s", exc)

    print(json.dumps({"report_json": str(output_file)}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()


__all__ = [
    "aggregate_eval_results",
    "main",
]


# Re-export for tests that need it without depending on private names.
def get_run_dir(output_jsonl: str | os.PathLike) -> Path:
    """Public alias for the internal run-dir resolver."""
    return _resolve_run_dir(Path(output_jsonl))
