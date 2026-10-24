from unittest.mock import Mock

from modal import Sandbox
from swebench.harness.modal_eval.run_evaluation_modal import ModalSandboxRuntime

from benchmarks.utils.modal_patches import _patch_modal_sandbox_filesystem


def test_runtime_writes_using_supported_filesystem_api(monkeypatch):
    # Only the remote Modal transport is mocked; exercise the patched runtime method.
    filesystem = Mock(spec=["write_text"])
    runtime = object.__new__(ModalSandboxRuntime)
    runtime.sandbox = Mock(spec=Sandbox, filesystem=filesystem)
    monkeypatch.setattr(
        ModalSandboxRuntime, "write_file", ModalSandboxRuntime.write_file
    )

    _patch_modal_sandbox_filesystem()
    _patch_modal_sandbox_filesystem()
    runtime.write_file("/root/patch.diff", "patch contents\n")

    filesystem.write_text.assert_called_once_with(
        "patch contents\n", "/root/patch.diff"
    )
