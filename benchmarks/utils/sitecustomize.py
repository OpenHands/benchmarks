"""
Site-wide hooks for benchmarks.

When running SWE-Bench evaluation on Modal, we want to capture exceptions that
happen before a `report.json` is written (e.g., sandbox creation failures). The
upstream harness only prints these exceptions, so the scoring step sees missing
logs and marks the instance as a generic error. This module monkey-patches
`run_instances_modal` to persist a minimal log/report for any exception result.

We also patch the scikit-learn install command used inside Modal sandboxes to
drop the deprecated `--no-use-pep517` flag (removed in pip>=25). That flag
breaks the sandbox image build before any logs are produced.

This file is imported automatically by Python when present on `sys.path`
(`PYTHONPATH` already includes `/workspace/benchmarks` in the evaluation job),
so no extra wiring is needed.
"""

from __future__ import annotations

import json
import time
import traceback


def _patch_modal_sklearn_install_flag() -> None:
    """
    pip>=25 removed `--no-use-pep517`, but the scikit-learn specs still pass it.
    When Modal builds the sandbox image, pip fails before tests ever run. Mutate
    the specs in-place to drop that flag for all scikit-learn versions.
    """
    try:
        # The constants module aliases SPECS_SKLEARN into MAP_REPO_VERSION_TO_SPECS,
        # so mutating the dict is sufficient as long as imports share the object.
        import swebench.harness.constants as consts
        import swebench.harness.constants.python as py_consts
    except Exception:
        return

    for version, spec in py_consts.SPECS_SKLEARN.items():
        install_cmd = spec.get("install", "")
        if "--no-use-pep517" not in install_cmd:
            continue

        cleaned = " ".join(install_cmd.replace("--no-use-pep517", "").split())
        py_consts.SPECS_SKLEARN[version]["install"] = cleaned

        repo_specs = consts.MAP_REPO_VERSION_TO_SPECS.get("scikit-learn/scikit-learn")
        if isinstance(repo_specs, dict):
            repo_specs[version] = py_consts.SPECS_SKLEARN[version]

    # Best-effort patch; stay silent if nothing needed or imports fail.
    return


def _patch_modal_sandbox_cgroup_retry() -> None:
    try:
        from swebench.harness.modal_eval import run_evaluation_modal as mod
    except Exception:
        return

    runtime_cls = getattr(mod, "ModalSandboxRuntime", None)
    if runtime_cls is None:
        return

    original_write_file = runtime_cls.write_file
    if getattr(original_write_file, "_benchmarks_retry_patch", False):
        return

    try:
        from modal.exception import FilesystemExecutionError
    except Exception:
        FilesystemExecutionError = Exception

    def write_file_with_retry(self, file_path: str, content: str):
        target_path = "/sys/fs/cgroup/cpu/cpu.shares"
        attempts = 5
        delay = 1.0
        path_str = str(file_path)
        for attempt in range(1, attempts + 1):
            try:
                return original_write_file(self, file_path, content)
            except Exception as exc:
                if path_str != target_path or not isinstance(
                    exc, FilesystemExecutionError
                ):
                    raise
                if attempt == attempts:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 10.0)

    write_file_with_retry._benchmarks_retry_patch = True
    runtime_cls.write_file = write_file_with_retry


def _apply_modal_logging_patch() -> None:
    _patch_modal_sklearn_install_flag()
    _patch_modal_sandbox_cgroup_retry()

    try:
        # Import inside the function so this file is harmless for non-SWE-Bench runs.
        from swebench.harness.docker_build import setup_logger
        from swebench.harness.modal_eval import run_evaluation_modal as mod
        from swebench.harness.modal_eval.run_evaluation_modal import (
            TestOutput,
            get_log_dir,
        )
        from swebench.harness.reporting import make_run_report
        from swebench.harness.test_spec.test_spec import make_test_spec
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
        max_attempts = 3
        attempt = 0
        backoff = 5.0
        try:
            import modal as modal_pkg

            client_closed_exc = getattr(
                getattr(modal_pkg, "exception", None), "ClientClosed", None
            )
        except Exception:
            client_closed_exc = None

        def is_client_closed_error(error: Exception) -> bool:
            if client_closed_exc is not None and isinstance(error, client_closed_exc):
                return True
            return "ClientClosed" in str(error)

        while True:
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

            if not run_test_specs:
                break

            attempt += 1
            client_closed_specs = []
            try:
                with mod.modal.enable_output():
                    with mod.app.run():
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
                                if is_client_closed_error(result):
                                    client_closed_specs.append((test_spec, result))
                                    continue
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
                if client_closed_specs:
                    if attempt < max_attempts:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 60.0)
                        continue
                    for test_spec, result in client_closed_specs:
                        pred = predictions[test_spec.instance_id]
                        log_dir = get_log_dir(pred, run_id, test_spec.instance_id)
                        if log_dir.exists():
                            continue
                        log_dir.mkdir(parents=True, exist_ok=True)
                        log_file = log_dir / "run_instance.log"
                        logger = setup_logger(
                            test_spec.instance_id, log_file, add_stdout=False
                        )
                        logger.error(
                            "Modal client closed during image build/sandbox create: %s",
                            result,
                        )
                        (log_dir / "patch.diff").write_text(
                            pred.get("model_patch", "")
                        )
                        report = {
                            test_spec.instance_id: {
                                "resolved": False,
                                "error": (
                                    "Modal client closed during image build/sandbox "
                                    f"create: {result}"
                                ),
                            }
                        }
                        (log_dir / "report.json").write_text(
                            json.dumps(report, indent=4)
                        )
                    break
            except Exception as exc:
                is_client_closed = is_client_closed_error(exc)

                if is_client_closed and attempt < max_attempts:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                    continue

                if is_client_closed:
                    for test_spec in run_test_specs:
                        pred = predictions[test_spec.instance_id]
                        log_dir = get_log_dir(pred, run_id, test_spec.instance_id)
                        if log_dir.exists():
                            continue
                        log_dir.mkdir(parents=True, exist_ok=True)
                        log_file = log_dir / "run_instance.log"
                        logger = setup_logger(
                            test_spec.instance_id, log_file, add_stdout=False
                        )
                        logger.error(
                            "Modal client closed during image build/sandbox create: %s",
                            exc,
                        )
                        (log_dir / "patch.diff").write_text(
                            pred.get("model_patch", "")
                        )
                        report = {
                            test_spec.instance_id: {
                                "resolved": False,
                                "error": f"Modal client closed: {exc}",
                            }
                        }
                        (log_dir / "report.json").write_text(
                            json.dumps(report, indent=4)
                        )
                    break

                raise

        # Always build the aggregate report (upstream behavior).
        make_run_report(predictions, full_dataset, run_id)

    # Apply the monkey patch once per interpreter.
    mod.run_instances_modal = run_instances_modal_with_logging
    try:
        # run_evaluation imports run_instances_modal by value, so update it too.
        import swebench.harness.run_evaluation as run_eval_mod

        run_eval_mod.run_instances_modal = run_instances_modal_with_logging
    except Exception:
        # If run_evaluation isn't available yet, skip—sitecustomize will have
        # already patched the modal module itself.
        pass
    try:
        # modal_eval re-exports run_instances_modal; update the package export too.
        import swebench.harness.modal_eval as modal_eval_pkg

        modal_eval_pkg.run_instances_modal = run_instances_modal_with_logging
    except Exception:
        # Keep best-effort behavior if the package import fails.
        pass


_apply_modal_logging_patch()
