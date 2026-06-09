"""Apptainer-based SWE-bench evaluation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from openhands.sdk import get_logger


logger = get_logger(__name__)


DEFAULT_APPTAINER_SANDBOX_ROOT = (
    Path.home() / ".cache" / "openhands" / "swebench-apptainer"
)


def _run(
    cmd: list[str],
    log_path: Path | None = None,
    timeout: int | None = None,
    apptainer_cache: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if apptainer_cache is not None:
        env["APPTAINER_CACHEDIR"] = str(apptainer_cache)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        env=env,
    )
    if log_path is not None:
        log_path.write_text(proc.stdout)
    return proc


def load_predictions(predictions_file: Path) -> dict[str, dict[str, Any]]:
    """Load SWE-bench predictions by instance id."""
    from swebench.harness.constants import KEY_INSTANCE_ID

    rows = []
    with predictions_file.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return {row[KEY_INSTANCE_ID]: row for row in rows}


def image_uri(instance: dict[str, Any]) -> str:
    """Return the official SWE-bench instance image URI."""
    from swebench.harness.test_spec.test_spec import make_test_spec

    spec = make_test_spec(instance, namespace="swebench")
    return "docker://" + spec.instance_image_key


def ensure_sandbox(
    instance: dict[str, Any],
    score_dir: Path,
    sandbox_root: Path,
    apptainer_cache: Path | None,
) -> Path:
    """Build or reuse an Apptainer sandbox for a SWE-bench instance."""
    from swebench.harness.constants import KEY_INSTANCE_ID

    instance_id = instance[KEY_INSTANCE_ID]
    sandbox = sandbox_root / instance_id
    if sandbox.exists():
        return sandbox

    tmp = sandbox.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building Apptainer sandbox for %s", instance_id)
    proc = _run(
        ["apptainer", "build", "--sandbox", str(tmp), image_uri(instance)],
        score_dir / f"{instance_id}.build.log",
        timeout=3600,
        apptainer_cache=apptainer_cache,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"apptainer build failed for {instance_id}; see "
            f"{score_dir / f'{instance_id}.build.log'}"
        )

    tmp.rename(sandbox)
    return sandbox


def apptainer_base_cmd(sandbox: Path, work_dir: Path) -> list[str]:
    """Return the base Apptainer exec command for a sandboxed test run."""
    return [
        "apptainer",
        "exec",
        "--no-mount",
        "hostfs",
        # Some clusters bind host /opt into every Apptainer container. Bind the
        # sandbox's /opt back over /opt so /opt/miniconda3 from SWE-bench images
        # remains visible.
        "--bind",
        f"{sandbox / 'opt'}:/opt",
        "--bind",
        f"{work_dir}:/mnt",
        "--writable",
        str(sandbox),
        "bash",
        "-lc",
    ]


def verify_sandbox(
    instance: dict[str, Any],
    sandbox: Path,
    work_dir: Path,
    apptainer_cache: Path | None,
) -> None:
    """Check that the SWE-bench conda environment is available in the sandbox."""
    from swebench.harness.constants import KEY_INSTANCE_ID

    cmd = apptainer_base_cmd(sandbox, work_dir) + [
        "source /opt/miniconda3/bin/activate && "
        "conda activate testbed && "
        "cd /testbed && "
        "python --version && "
        "command -v python && "
        "git rev-parse HEAD"
    ]
    proc = _run(
        cmd,
        work_dir / "sandbox_verify.log",
        timeout=120,
        apptainer_cache=apptainer_cache,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sandbox verification failed for {instance[KEY_INSTANCE_ID]}; "
            f"see {work_dir / 'sandbox_verify.log'}"
        )


def score_instance(
    instance: dict[str, Any],
    prediction: dict[str, Any],
    score_dir: Path,
    sandbox_root: Path,
    timeout_seconds: int,
    apptainer_cache: Path | None,
) -> dict[str, Any]:
    """Apply a prediction and score one SWE-bench instance in Apptainer."""
    from swebench.harness.constants import (
        APPLY_PATCH_FAIL,
        KEY_INSTANCE_ID,
        KEY_PREDICTION,
        TESTS_TIMEOUT,
    )
    from swebench.harness.grading import get_eval_report
    from swebench.harness.test_spec.test_spec import make_test_spec

    instance_id = instance[KEY_INSTANCE_ID]
    work_dir = score_dir / instance_id
    work_dir.mkdir(parents=True, exist_ok=True)

    report_path = work_dir / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())

    patch = prediction.get(KEY_PREDICTION) or ""
    if not patch.strip():
        report = {
            instance_id: {
                "patch_is_None": prediction.get(KEY_PREDICTION) is None,
                "patch_exists": False,
                "patch_successfully_applied": False,
                "resolved": False,
                "skipped_empty_patch": True,
            }
        }
        report_path.write_text(json.dumps(report, indent=2))
        return report

    test_spec = make_test_spec(instance)
    sandbox = ensure_sandbox(instance, score_dir, sandbox_root, apptainer_cache)
    (work_dir / "patch.diff").write_text(patch)
    (work_dir / "eval.sh").write_text(test_spec.eval_script)
    verify_sandbox(instance, sandbox, work_dir, apptainer_cache)

    shell = f"""
set -uo pipefail
cd /testbed
git config --global --add safe.directory /testbed || true
git reset --hard HEAD
git clean -fd
applied=0
apply_output=/mnt/apply_patch.log
: > "$apply_output"
for cmd in "git apply --verbose /mnt/patch.diff" "git apply --verbose --reject /mnt/patch.diff" "patch --batch --fuzz=5 -p1 -i /mnt/patch.diff"; do
  echo "$cmd" >> "$apply_output"
  if bash -lc "$cmd" >> "$apply_output" 2>&1; then
    applied=1
    break
  fi
done
if [ "$applied" != "1" ]; then
  {{
    echo "{APPLY_PATCH_FAIL}"
    cat "$apply_output"
  }} > /mnt/test_output.txt
  exit 0
fi
git -c core.fileMode=false diff > /mnt/git_diff_before.diff 2>&1 || true
timeout {timeout_seconds} /bin/bash /mnt/eval.sh > /mnt/test_output.txt 2>&1
status=$?
git -c core.fileMode=false diff > /mnt/git_diff_after.diff 2>&1 || true
if [ "$status" = "124" ]; then
  echo "{TESTS_TIMEOUT}" >> /mnt/test_output.txt
fi
exit 0
"""
    proc = _run(
        apptainer_base_cmd(sandbox, work_dir) + [shell],
        work_dir / "apptainer_exec.log",
        timeout=timeout_seconds + 300,
        apptainer_cache=apptainer_cache,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"apptainer exec failed for {instance_id}; "
            f"see {work_dir / 'apptainer_exec.log'}"
        )

    report = get_eval_report(
        test_spec=test_spec,
        prediction=prediction,
        test_log_path=str(work_dir / "test_output.txt"),
        include_tests_status=True,
    )
    report_path.write_text(json.dumps(report, indent=2))
    return report


def run_swebench_evaluation_apptainer(
    predictions_file: Path,
    report_file: Path,
    dataset: str,
    split: str,
    timeout_seconds: int,
    workers: int,
    score_dir: Path | None = None,
    sandbox_root: Path = DEFAULT_APPTAINER_SANDBOX_ROOT,
    apptainer_cache: Path | None = None,
) -> None:
    """Run SWE-bench evaluation locally with Apptainer."""
    from datasets import load_dataset
    from swebench.harness.constants import KEY_INSTANCE_ID

    if workers != 1:
        logger.warning(
            "Apptainer evaluation currently runs sequentially; ignoring workers=%s",
            workers,
        )

    predictions = load_predictions(predictions_file)
    wanted = set(predictions)
    if score_dir is None:
        score_dir = predictions_file.parent / "apptainer_eval"
    score_dir.mkdir(parents=True, exist_ok=True)
    sandbox_root.mkdir(parents=True, exist_ok=True)

    instances = [
        instance
        for instance in load_dataset(dataset, split=split)
        if instance[KEY_INSTANCE_ID] in wanted
    ]

    reports: dict[str, Any] = {}
    for instance in instances:
        instance_id = instance[KEY_INSTANCE_ID]
        logger.info("Scoring %s with Apptainer", instance_id)
        report = score_instance(
            instance=instance,
            prediction=predictions[instance_id],
            score_dir=score_dir,
            sandbox_root=sandbox_root,
            timeout_seconds=timeout_seconds,
            apptainer_cache=apptainer_cache,
        )
        reports.update(report)
        logger.info(
            "%s: resolved=%s patch_applied=%s",
            instance_id,
            reports[instance_id].get("resolved"),
            reports[instance_id].get("patch_successfully_applied"),
        )

    resolved_ids = sorted(
        instance_id for instance_id, report in reports.items() if report.get("resolved")
    )
    unresolved_ids = sorted(set(reports) - set(resolved_ids))
    report_data = {
        "dataset": dataset,
        "split": split,
        "predictions": str(predictions_file),
        "score_dir": str(score_dir),
        "total": len(reports),
        "resolved": len(resolved_ids),
        "unresolved": len(unresolved_ids),
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "reports": reports,
    }
    report_file.write_text(json.dumps(report_data, indent=2))
    logger.info("Wrote Apptainer SWE-bench report to %s", report_file)
