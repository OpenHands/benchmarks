"""
Shared Modal patch helpers for host and in-image sitecustomize.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

_MODAL_SITECUSTOMIZE_INJECTED = False
DEFAULT_AGENT_IMAGE = "ghcr.io/openhands/eval-agent-server"
DEFAULT_BUILD_TARGET = "source-minimal"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _make_emit(stderr: bool):
    if stderr:

        def emit(message: str) -> None:
            print(message, file=sys.stderr, flush=True)

    else:

        def emit(message: str) -> None:
            print(message)

    return emit


def _get_sdk_short_sha() -> str:
    """
    Resolve SDK short SHA from the benchmarks repo when available, otherwise
    fall back to environment variables for the Modal function image.
    """
    try:
        from benchmarks.utils.version import SDK_SHORT_SHA as version_sdk_short_sha

        return version_sdk_short_sha
    except Exception:
        return os.getenv("SDK_SHORT_SHA", "").strip() or "unknown"


def _get_agent_server_image_repo() -> str:
    return os.getenv("EVAL_AGENT_SERVER_IMAGE", DEFAULT_AGENT_IMAGE).strip() or DEFAULT_AGENT_IMAGE


def _get_build_target() -> str:
    return (
        os.getenv("SWEBENCH_IMAGE_TARGET")
        or os.getenv("SWEBENCH_BUILD_TARGET")
        or DEFAULT_BUILD_TARGET
    )


def _get_custom_tag_from_instance_id(instance_id: str) -> str:
    try:
        repo, name = instance_id.split("__", 1)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to compute SWE-bench image tag; unexpected instance id: {instance_id}"
        ) from exc
    return f"sweb.eval.x86_64.{repo}_1776_{name}".lower()


def _build_prebuilt_image_tag(test_spec) -> str:
    instance_id = getattr(test_spec, "instance_id", None)
    if not instance_id:
        raise RuntimeError("TestSpec missing instance_id; cannot select Modal image")

    sdk_short_sha = _get_sdk_short_sha()
    if sdk_short_sha in ("", "unknown", None):
        raise RuntimeError(
            "SDK short SHA is unavailable. Set SDK_SHORT_SHA or ensure the "
            "benchmarks repository has an initialized SDK submodule."
        )

    target = _get_build_target()
    suffix = f"-{target}" if target and target != "binary" else ""
    custom_tag = _get_custom_tag_from_instance_id(instance_id)
    agent_repo = _get_agent_server_image_repo()
    return f"{agent_repo}:{sdk_short_sha}-{custom_tag}{suffix}"


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
    """Retry cgroup writes to avoid transient Modal filesystem errors."""
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


def _patch_modal_prebuilt_images(log_errors: bool = False, stderr: bool = False) -> None:
    """Use prebuilt SWE-Bench images in Modal instead of rebuilding per instance."""
    try:
        from swebench.harness.modal_eval import run_evaluation_modal as mod
    except Exception as exc:
        if log_errors:
            _log(
                f"[benchmarks] modal sitecustomize: failed to import modal_eval: {exc}"
            )
        return

    runtime_cls = getattr(mod, "ModalSandboxRuntime", None)
    if runtime_cls is None:
        if log_errors:
            _log("[benchmarks] modal sitecustomize: ModalSandboxRuntime missing")
        return

    original_get_instance_image = getattr(runtime_cls, "get_instance_image", None)
    if original_get_instance_image is None:
        if log_errors:
            _log("[benchmarks] modal sitecustomize: get_instance_image missing")
        return
    if getattr(original_get_instance_image, "_benchmarks_prebuilt_patch", False):
        return

    emit = _make_emit(stderr)

    def get_instance_image_from_registry(test_spec):
        import modal

        instance_id = getattr(test_spec, "instance_id", "unknown")
        try:
            image_tag = _build_prebuilt_image_tag(test_spec)
        except Exception as exc:
            emit(
                "[benchmarks] Modal image spec failed to compute tag for "
                f"{instance_id}: {exc}"
            )
            raise

        emit(
            "[benchmarks] Modal image spec using prebuilt image "
            f"{image_tag} for {instance_id}"
        )
        try:
            image = modal.Image.from_registry(image_tag)
        except Exception as exc:
            emit(
                "[benchmarks] Failed to load Modal image from registry "
                f"{image_tag}: {exc}"
            )
            raise

        # Upstream expects /testbed as the working directory when running evals.
        return image.workdir("/testbed/")

    get_instance_image_from_registry._benchmarks_prebuilt_patch = True
    runtime_cls.get_instance_image = staticmethod(get_instance_image_from_registry)
    if log_errors:
        _log("[benchmarks] modal sitecustomize: applied prebuilt image patch")


def _patch_modal_sandbox_timing(log_errors: bool = False, stderr: bool = False) -> None:
    """Log sandbox creation timing to pinpoint Modal startup delays."""
    try:
        from swebench.harness.modal_eval import run_evaluation_modal as mod
    except Exception as exc:
        if log_errors:
            _log(
                f"[benchmarks] modal sitecustomize: failed to import modal_eval: {exc}"
            )
        return

    runtime_cls = getattr(mod, "ModalSandboxRuntime", None)
    if runtime_cls is None:
        if log_errors:
            _log("[benchmarks] modal sitecustomize: ModalSandboxRuntime missing")
        return

    original_get_sandbox = runtime_cls._get_sandbox
    if getattr(original_get_sandbox, "_benchmarks_timing_patch", False):
        return

    emit = _make_emit(stderr)

    def get_sandbox_with_timing(self, timeout: int | None = None):
        instance_id = getattr(getattr(self, "test_spec", None), "instance_id", "unknown")
        start = time.time()
        emit(
            f"[benchmarks] Modal sandbox create start for {instance_id} "
            f"(timeout={timeout})"
        )
        try:
            return original_get_sandbox(self, timeout)
        finally:
            elapsed = time.time() - start
            emit(
                f"[benchmarks] Modal sandbox create end for {instance_id} "
                f"(elapsed={elapsed:.2f}s)"
            )

    get_sandbox_with_timing._benchmarks_timing_patch = True
    runtime_cls._get_sandbox = get_sandbox_with_timing
    if log_errors:
        _log("[benchmarks] modal sitecustomize: applied sandbox timing patch")


def _patch_modal_runtime_debug(log_errors: bool = False, stderr: bool = False) -> None:
    """Log Modal runtime init and critical exec timings for debugging."""
    try:
        from swebench.harness.modal_eval import run_evaluation_modal as mod
    except Exception as exc:
        if log_errors:
            _log(f"[benchmarks] modal sitecustomize: failed to import modal_eval: {exc}")
        return

    runtime_cls = getattr(mod, "ModalSandboxRuntime", None)
    if runtime_cls is None:
        if log_errors:
            _log("[benchmarks] modal sitecustomize: ModalSandboxRuntime missing")
        return

    emit = _make_emit(stderr)

    original_init = runtime_cls.__init__
    if not getattr(original_init, "_benchmarks_runtime_init_patch", False):

        def init_with_logging(self, test_spec, timeout: int | None = None, verbose=True):
            instance_id = getattr(test_spec, "instance_id", "unknown")
            emit(
                f"[benchmarks] Modal runtime init start for {instance_id} "
                f"(timeout={timeout})"
            )
            start = time.time()
            try:
                return original_init(self, test_spec, timeout, verbose)
            finally:
                elapsed = time.time() - start
                emit(
                    f"[benchmarks] Modal runtime init end for {instance_id} "
                    f"(elapsed={elapsed:.2f}s)"
                )

        init_with_logging._benchmarks_runtime_init_patch = True
        runtime_cls.__init__ = init_with_logging

    original_exec = runtime_cls.exec
    if not getattr(original_exec, "_benchmarks_runtime_exec_patch", False):

        def exec_with_logging(self, command: str):
            instance_id = getattr(getattr(self, "test_spec", None), "instance_id", "unknown")
            label = None
            if "/root/eval.sh" in command:
                label = "eval"
            elif "git apply" in command or "patch --batch" in command:
                label = "apply_patch"

            if label:
                emit(f"[benchmarks] Modal exec start for {instance_id} ({label})")
                start = time.time()
                output, returncode = original_exec(self, command)
                elapsed = time.time() - start
                emit(
                    f"[benchmarks] Modal exec end for {instance_id} ({label}) "
                    f"(elapsed={elapsed:.2f}s, returncode={returncode})"
                )
                return output, returncode

            return original_exec(self, command)

        exec_with_logging._benchmarks_runtime_exec_patch = True
        runtime_cls.exec = exec_with_logging

    if log_errors:
        _log("[benchmarks] modal sitecustomize: applied runtime debug patch")


def _patch_modal_function_timeout(
    timeout_seconds: int = 4 * 60 * 60, log_errors: bool = False
) -> None:
    """Raise Modal function timeout and emit per-instance logs in Modal."""
    try:
        from swebench.harness.modal_eval import run_evaluation_modal as mod
    except Exception as exc:
        if log_errors:
            _log(f"[benchmarks] modal sitecustomize: failed to import modal_eval: {exc}")
        return

    run_fn = getattr(mod, "run_instance_modal", None)
    if run_fn is None:
        if log_errors:
            _log("[benchmarks] modal sitecustomize: run_instance_modal missing")
        return
    if getattr(run_fn, "_benchmarks_timeout_patch", False):
        return

    raw_f = getattr(getattr(run_fn, "info", None), "raw_f", None)
    if raw_f is None:
        if log_errors:
            _log("[benchmarks] modal sitecustomize: run_instance_modal raw_f missing")
        return

    image = getattr(getattr(run_fn, "spec", None), "image", None)
    if image is None:
        image = getattr(mod, "swebench_image", None)

    def run_instance_modal_with_logging(test_spec, pred, run_id, timeout=None):
        instance_id = getattr(test_spec, "instance_id", None) or pred.get(
            "instance_id", "unknown"
        )
        effective_timeout = timeout
        if timeout is None or timeout < timeout_seconds:
            effective_timeout = timeout_seconds
            print(
                "[benchmarks] Modal function overriding timeout "
                f"instance={instance_id} from {timeout} to {effective_timeout}",
                file=sys.stderr,
                flush=True,
            )
        start = time.time()
        print(
            "[benchmarks] Modal function start "
            f"instance={instance_id} run_id={run_id} timeout={effective_timeout}",
            file=sys.stderr,
            flush=True,
        )
        try:
            result = raw_f(test_spec, pred, run_id, effective_timeout)
        except Exception as exc:
            elapsed = time.time() - start
            print(
                "[benchmarks] Modal function error "
                f"instance={instance_id} elapsed={elapsed:.2f}s error={exc}",
                file=sys.stderr,
                flush=True,
            )
            raise
        elapsed = time.time() - start
        status = "errored" if getattr(result, "errored", False) else "ok"
        print(
            "[benchmarks] Modal function end "
            f"instance={instance_id} elapsed={elapsed:.2f}s status={status}",
            file=sys.stderr,
            flush=True,
        )
        return result

    try:
        patched_fn = mod.app.function(
            image=image,
            timeout=timeout_seconds,
            include_source=True,
            serialized=True,
            name="run_instance_modal",
        )(run_instance_modal_with_logging)
    except Exception as exc:
        if log_errors:
            _log(
                "[benchmarks] modal sitecustomize: failed to patch timeout: "
                f"{exc}"
            )
        return

    patched_fn._benchmarks_timeout_patch = True
    mod.run_instance_modal = patched_fn
    if log_errors:
        _log(
            "[benchmarks] modal sitecustomize: patched function timeout "
            f"to {timeout_seconds}s"
        )


def _inject_modal_sitecustomize() -> None:
    """Inject modal_sitecustomize into the Modal function image."""
    global _MODAL_SITECUSTOMIZE_INJECTED

    if _MODAL_SITECUSTOMIZE_INJECTED:
        return

    try:
        from pathlib import Path

        from swebench.harness.modal_eval import run_evaluation_modal as mod
    except Exception:
        return

    patch_path = Path(__file__).with_name("modal_sitecustomize.py")
    if not patch_path.exists():
        return

    patches_path = Path(__file__).with_name("modal_patches.py")

    run_fn = getattr(mod, "run_instance_modal", None)
    if run_fn is None or not hasattr(run_fn, "spec"):
        return

    image = run_fn.spec.image

    # Rebuild from the base swebench image so add_local_file mounts (from the
    # original function definition) are converted to copies. Modal rejects
    # adding build steps after mount layers.
    base_image = getattr(mod, "swebench_image", None)
    entry_local = getattr(mod, "LOCAL_SANDBOX_ENTRYPOINT_PATH", None)
    entry_remote = getattr(mod, "REMOTE_SANDBOX_ENTRYPOINT_PATH", None)
    if base_image is not None and entry_local is not None and entry_remote is not None:
        image = base_image.add_local_file(
            Path(entry_local),
            str(entry_remote),
            copy=True,
        )

    patched_image = image.add_local_file(
        patch_path,
        "/root/sitecustomize.py",
        copy=True,
    )

    if patches_path.exists():
        patched_image = patched_image.add_local_file(
            patches_path,
            "/root/modal_patches.py",
            copy=True,
        )

    env_vars = {"PYTHONPATH": "/root"}
    try:
        from benchmarks.utils.version import SDK_SHA, SDK_SHORT_SHA

        env_vars["SDK_SHA"] = SDK_SHA
        env_vars["SDK_SHORT_SHA"] = SDK_SHORT_SHA
    except Exception:
        if os.getenv("SDK_SHA"):
            env_vars["SDK_SHA"] = os.getenv("SDK_SHA")
        env_vars["SDK_SHORT_SHA"] = _get_sdk_short_sha()

    env_vars["EVAL_AGENT_SERVER_IMAGE"] = _get_agent_server_image_repo()
    env_vars["SWEBENCH_IMAGE_TARGET"] = _get_build_target()

    patched_image = patched_image.env(env_vars)

    run_fn.spec.image = patched_image
    mod.swebench_image = patched_image
    _MODAL_SITECUSTOMIZE_INJECTED = True
    _log("benchmarks injected modal sitecustomize into run_instance_modal image")


def _patch_run_instances_modal_logging() -> None:
    """Persist logs/reports for Modal exceptions before TestOutput is returned."""
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
                        emit = _make_emit(stderr=False)
                        submit_ids = [spec.instance_id for spec in run_test_specs]
                        emit(
                            f"[benchmarks] Modal starmap submit {len(submit_ids)} "
                            f"instances: {', '.join(submit_ids)}"
                        )
                        starmap_start = time.time()
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
                        starmap_elapsed = time.time() - starmap_start
                        emit(
                            f"[benchmarks] Modal starmap completed in "
                            f"{starmap_elapsed:.2f}s"
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


def apply_host_patches() -> None:
    _patch_modal_sklearn_install_flag()
    _patch_modal_sandbox_cgroup_retry()
    _patch_modal_prebuilt_images()
    # Inject sitecustomize before re-registering the Modal function so the
    # patched image (with env + sitecustomize) is baked into the function spec.
    _inject_modal_sitecustomize()
    _patch_modal_sandbox_timing(log_errors=True, stderr=True)
    _patch_modal_runtime_debug(log_errors=True, stderr=True)
    _patch_modal_function_timeout(log_errors=True)
    _patch_run_instances_modal_logging()


def apply_image_patches() -> None:
    _log("[benchmarks] modal sitecustomize imported")
    _patch_modal_prebuilt_images(log_errors=True, stderr=True)
    _patch_modal_sandbox_timing(log_errors=True, stderr=True)
    _patch_modal_runtime_debug(log_errors=True, stderr=True)
