"""
Sitecustomize injected into the Modal function image for SWE-bench runs.

This file is copied into the Modal function container and imported automatically
by Python (via sitecustomize) to patch the modal_eval runtime with extra timing
logs and a libmamba solver setup.
"""

from __future__ import annotations

import sys
import time


def _apply_modal_image_patch() -> None:
    print("[benchmarks] modal sitecustomize imported", file=sys.stderr, flush=True)

    try:
        from swebench.harness.modal_eval import run_evaluation_modal as mod
    except Exception as exc:
        print(
            f"[benchmarks] modal sitecustomize: failed to import modal_eval: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return

    runtime_cls = getattr(mod, "ModalSandboxRuntime", None)
    if runtime_cls is None:
        print(
            "[benchmarks] modal sitecustomize: ModalSandboxRuntime missing",
            file=sys.stderr,
            flush=True,
        )
        return

    original_get_instance_image = getattr(runtime_cls, "get_instance_image", None)
    if original_get_instance_image is None:
        print(
            "[benchmarks] modal sitecustomize: get_instance_image missing",
            file=sys.stderr,
            flush=True,
        )
        return

    if not getattr(original_get_instance_image, "_benchmarks_libmamba_patch", False):

        def get_instance_image_with_libmamba(test_spec):
            import modal
            from pathlib import Path

            start = time.time()
            instance_id = getattr(test_spec, "instance_id", "unknown")
            print(
                f"[benchmarks] Modal image spec start for {instance_id}",
                file=sys.stderr,
                flush=True,
            )

            env_script = test_spec.setup_env_script
            env_script = env_script.replace(
                "conda activate testbed && python -m pip install -r $HOME/requirements.txt",
                "conda activate testbed && python -m pip install --trusted-host "
                "pypi-mirror.modal.local -r $HOME/requirements.txt",
            )
            repo_script = test_spec.install_repo_script

            remote_env_script_path = "/root/setup_env.sh"
            remote_repo_script_path = "/root/setup_repo.sh"

            Path(remote_env_script_path).write_text(env_script)
            Path(remote_repo_script_path).write_text(repo_script)

            image = (
                modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
                .run_commands("apt update")
                .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
                .apt_install(
                    "wget",
                    "git",
                    "build-essential",
                    "libffi-dev",
                    "libtiff-dev",
                    "jq",
                    "curl",
                    "locales",
                    "locales-all",
                    "tzdata",
                )
                .run_commands(
                    "wget 'https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh' -O miniconda.sh",
                    "bash miniconda.sh -b -p /opt/miniconda3",
                    "echo 'export PATH=/opt/miniconda3/bin:$PATH' >> ~/.bashrc",
                    "/opt/miniconda3/bin/conda init --all",
                    "/opt/miniconda3/bin/conda config --append channels conda-forge",
                    "/bin/bash -c '/opt/miniconda3/bin/conda install -n base -y conda-libmamba-solver "
                    "|| echo \"conda-libmamba-solver install failed; continuing\"'",
                    "/bin/bash -c '/opt/miniconda3/bin/conda config --set solver libmamba "
                    "|| echo \"conda libmamba solver unavailable; continuing\"'",
                    "adduser --disabled-password --gecos 'dog' nonroot",
                )
                .add_local_file(
                    Path(remote_env_script_path), remote_env_script_path, copy=True
                )
                .add_local_file(
                    Path(remote_repo_script_path), remote_repo_script_path, copy=True
                )
                .run_commands(
                    f"chmod +x {remote_env_script_path}",
                    f"/bin/bash -c 'source ~/.bashrc && {remote_env_script_path}'",
                    "echo 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed' >> /root/.bashrc",
                    f"/bin/bash {remote_repo_script_path}",
                )
                .workdir("/testbed/")
            )

            elapsed = time.time() - start
            print(
                f"[benchmarks] Modal image spec end for {instance_id} "
                f"(elapsed={elapsed:.2f}s)",
                file=sys.stderr,
                flush=True,
            )
            return image

        get_instance_image_with_libmamba._benchmarks_libmamba_patch = True
        runtime_cls.get_instance_image = staticmethod(get_instance_image_with_libmamba)
        print(
            "[benchmarks] modal sitecustomize: applied libmamba patch",
            file=sys.stderr,
            flush=True,
        )

    original_get_sandbox = runtime_cls._get_sandbox
    if not getattr(original_get_sandbox, "_benchmarks_timing_patch", False):

        def get_sandbox_with_timing(self, timeout: int | None = None):
            instance_id = getattr(
                getattr(self, "test_spec", None), "instance_id", "unknown"
            )
            start = time.time()
            print(
                f"[benchmarks] Modal sandbox create start for {instance_id} "
                f"(timeout={timeout})",
                file=sys.stderr,
                flush=True,
            )
            try:
                return original_get_sandbox(self, timeout)
            finally:
                elapsed = time.time() - start
                print(
                    f"[benchmarks] Modal sandbox create end for {instance_id} "
                    f"(elapsed={elapsed:.2f}s)",
                    file=sys.stderr,
                    flush=True,
                )

        get_sandbox_with_timing._benchmarks_timing_patch = True
        runtime_cls._get_sandbox = get_sandbox_with_timing
        print(
            "[benchmarks] modal sitecustomize: applied sandbox timing patch",
            file=sys.stderr,
            flush=True,
        )


_apply_modal_image_patch()
