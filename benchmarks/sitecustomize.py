"""
Site-wide hooks for benchmarks.

When running SWE-Bench evaluation on Modal, we want to capture exceptions that
happen before a `report.json` is written (e.g., sandbox creation failures). The
upstream harness only prints these exceptions, so the scoring step sees missing
logs and marks the instance as a generic error. This module monkey-patches
`run_instances_modal` to persist a minimal log/report for any exception result.

This file is imported automatically by Python when present on `sys.path`
(`PYTHONPATH` already includes `/workspace/benchmarks` in the evaluation job),
so no extra wiring is needed.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path


def _apply_modal_logging_patch() -> None:
    try:
        # Import inside the function so this file is harmless for non-SWE-Bench runs.
        from swebench.harness.modal_eval import run_evaluation_modal as mod
        from swebench.harness.modal_eval.run_evaluation_modal import (
            TestOutput,
            get_log_dir,
        )
        from swebench.harness.reporting import make_run_report
        from swebench.harness.test_spec.test_spec import make_test_spec
        from swebench.harness.docker_build import setup_logger
    except Exception:
        # If swebench isn't installed, bail out quietly.
        return

    def run_instances_modal_with_logging(
        predictions: dict,
        instances: list,
        full_dataset: list,
        run_id: str,
        timeout: int,
    ):
        """
        Wrap the upstream `run_instances_modal` to persist logs for exceptions.

        If Modal returns an exception (e.g., sandbox creation failure), we now
        write run_instance.log + report.json so scoring can surface the error.
        """
        test_specs = list(map(make_test_spec, instances))

        with mod.modal.enable_output():
            with mod.app.run():
                run_test_specs = []

                # Skip any instances that already have logs.
                for test_spec in test_specs:
                    log_dir = get_log_dir(
                        predictions[test_spec.instance_id],
                        run_id,
                        test_spec.instance_id,
                    )
                    if log_dir.exists():
                        continue
                    run_test_specs.append(test_spec)

                if run_test_specs:
                    results = mod.run_instance_modal.starmap(
                        [
                            (
                                test_spec,
                                predictions[test_spec.instance_id],
                                run_id,
                                timeout,
                            )
                            for test_spec in run_test_specs
                        ],
                        return_exceptions=True,
                    )

                    for test_spec, result in zip(run_test_specs, results):
                        pred = predictions[test_spec.instance_id]
                        log_dir = get_log_dir(pred, run_id, test_spec.instance_id)
                        log_dir.mkdir(parents=True, exist_ok=True)

                        if isinstance(result, TestOutput):
                            # Normal path: write logs exactly as upstream does.
                            with open(log_dir / "run_instance.log", "w") as f:
                                f.write(result.run_instance_log)
                            with open(log_dir / "test_output.txt", "w") as f:
                                f.write(result.test_output)
                            with open(log_dir / "patch.diff", "w") as f:
                                f.write(result.patch_diff)
                            if result.report_json_str:
                                try:
                                    parsed = json.loads(result.report_json_str)
                                    (log_dir / "report.json").write_text(
                                        json.dumps(parsed, indent=4)
                                    )
                                except Exception:
                                    # Best-effort write if JSON is malformed.
                                    (log_dir / "report.json").write_text(
                                        result.report_json_str
                                    )
                        else:
                            # Exception path: persist a minimal log + report so scoring sees it.
                            log_file = log_dir / "run_instance.log"
                            logger = setup_logger(
                                test_spec.instance_id, log_file, add_stdout=False
                            )
                            logger.error(
                                "Modal run failed before producing TestOutput: %s",
                                result,
                            )
                            logger.error(
                                "Traceback:\n%s",
                                "".join(traceback.format_exception(result)),
                            )

                            # Save the attempted patch for debugging.
                            (log_dir / "patch.diff").write_text(
                                pred.get("model_patch", "")
                            )

                            error_msg = f"Modal error: {result}"
                            report = {
                                test_spec.instance_id: {
                                    "resolved": False,
                                    "error": error_msg,
                                }
                            }
                            (log_dir / "report.json").write_text(
                                json.dumps(report, indent=4)
                            )

        # Always build the aggregate report (upstream behavior).
        make_run_report(predictions, full_dataset, run_id)

    # Apply the monkey patch once per interpreter.
    mod.run_instances_modal = run_instances_modal_with_logging


_apply_modal_logging_patch()
