"""Tests for SWE-bench Apptainer image build fallback."""

from pathlib import Path
from types import SimpleNamespace

from benchmarks.swebench import apptainer_build, run_infer as swebench_run_infer
from benchmarks.utils.models import EvalInstance


class FakeApptainerWorkspace:
    """Capture ApptainerWorkspace constructor arguments."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _evaluation():
    metadata = SimpleNamespace(
        workspace_type="apptainer",
        agent_type="default",
        env_setup_commands=[],
    )
    evaluation = object.__new__(swebench_run_infer.SWEBenchEvaluation)
    object.__setattr__(evaluation, "metadata", metadata)
    return evaluation


def test_unsupported_apptainer_build_target_returns_error():
    output = apptainer_build.build_apptainer_agent_image(
        base_image="docker.io/swebench/example:latest",
        custom_tag="example",
        target="binary",
    )

    assert output.tags == []
    assert output.error is not None
    assert "source-minimal" in output.error


def test_apptainer_workspace_uses_registry_image_when_available(monkeypatch):
    monkeypatch.setattr(swebench_run_infer, "remote_image_exists", lambda image: True)
    monkeypatch.setattr(
        swebench_run_infer,
        "ApptainerWorkspace",
        FakeApptainerWorkspace,
    )

    workspace = _evaluation().prepare_workspace(
        EvalInstance(id="django__django-12345", data={})
    )

    assert "server_image" in workspace.kwargs
    assert "sif_file" not in workspace.kwargs


def test_apptainer_workspace_builds_local_sandbox_when_registry_image_missing(
    monkeypatch,
):
    built = {}

    def fake_build(**kwargs):
        built.update(kwargs)
        return Path("/tmp/local-agent.sandbox")

    monkeypatch.setattr(swebench_run_infer, "remote_image_exists", lambda image: False)
    monkeypatch.setattr(
        swebench_run_infer,
        "ensure_apptainer_agent_image",
        fake_build,
    )
    monkeypatch.setattr(
        swebench_run_infer,
        "ApptainerWorkspace",
        FakeApptainerWorkspace,
    )

    workspace = _evaluation().prepare_workspace(
        EvalInstance(id="django__django-12345", data={})
    )

    assert workspace.kwargs["sif_file"] == "/tmp/local-agent.sandbox"
    assert "server_image" not in workspace.kwargs
    assert built["base_image"].startswith("docker.io/swebench/")
    assert built["custom_tag"] == "sweb.eval.x86_64.django_1776_django-12345"
    assert built["target"] == "source-minimal"
