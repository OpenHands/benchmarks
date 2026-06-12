"""Tests for SWE-bench Apptainer image build fallback."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmarks.swebench import (
    apptainer_build,
    build_images,
    run_infer as swebench_run_infer,
)
from benchmarks.utils.models import EvalInstance


class FakeApptainerWorkspace:
    """Capture ApptainerWorkspace constructor arguments."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _evaluation():
    metadata = SimpleNamespace(
        workspace_type="apptainer",
        agent_type="default",
        env_setup_commands=[],
        llm=SimpleNamespace(custom_tokenizer=None),
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


def test_apptainer_definition_installs_transformers_for_token_counting():
    definition = apptainer_build._definition_file_content(
        base_image="docker.io/swebench/example:latest",
        git_sha="abc123",
        git_ref="main",
        wrap_swebench_deps=False,
        uv_path=Path("/usr/local/bin/uv"),
        uvx_path=None,
    )

    assert 'uv pip install --python /agent-server/.venv/bin/python "transformers' in (
        definition
    )


def test_failed_forced_apptainer_rebuild_keeps_existing_sif(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENHANDS_APPTAINER_BUILD_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENHANDS_APPTAINER_FORCE_BUILD", "1")
    monkeypatch.setattr(
        apptainer_build,
        "_get_sdk_submodule_info",
        lambda: ("main", "abcdef123456", ""),
    )
    monkeypatch.setattr(apptainer_build, "dockerfile_content_hash", lambda: "hash")
    monkeypatch.setattr(
        apptainer_build.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"apptainer", "uv", "uvx"} else None,
    )
    monkeypatch.setattr(
        apptainer_build.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=42, stdout="build failed"),
    )

    existing_image = apptainer_build.apptainer_agent_image_path("tag")
    existing_image.parent.mkdir(parents=True, exist_ok=True)
    existing_image.write_text("old working image")

    output = apptainer_build.build_apptainer_agent_image(
        base_image="docker.io/swebench/example:latest",
        custom_tag="tag",
    )

    assert output.error == "Apptainer build failed with exit code 42"
    assert existing_image.read_text() == "old working image"


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

    assert isinstance(workspace, FakeApptainerWorkspace)
    assert "server_image" in workspace.kwargs
    assert "sif_file" not in workspace.kwargs
    assert workspace.kwargs["extra_bind_mounts"] == []


def test_apptainer_workspace_binds_existing_custom_tokenizer(monkeypatch, tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    evaluation = _evaluation()
    evaluation.metadata.llm.custom_tokenizer = str(tokenizer_dir)

    monkeypatch.setattr(swebench_run_infer, "remote_image_exists", lambda image: True)
    monkeypatch.setattr(
        swebench_run_infer,
        "ApptainerWorkspace",
        FakeApptainerWorkspace,
    )

    workspace = evaluation.prepare_workspace(
        EvalInstance(id="django__django-12345", data={})
    )

    assert isinstance(workspace, FakeApptainerWorkspace)
    assert workspace.kwargs["extra_bind_mounts"] == [
        f"{tokenizer_dir}:{tokenizer_dir}:ro"
    ]


def test_apptainer_workspace_builds_local_sif_when_registry_image_missing(
    monkeypatch,
):
    built = {}

    def fake_build(**kwargs):
        built.update(kwargs)
        return Path("/tmp/local-agent.sif")

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

    assert isinstance(workspace, FakeApptainerWorkspace)
    assert workspace.kwargs["sif_file"] == "/tmp/local-agent.sif"
    assert "server_image" not in workspace.kwargs
    assert workspace.kwargs["extra_bind_mounts"] == []
    assert built["base_image"].startswith("docker.io/swebench/")
    assert built["custom_tag"] == "sweb.eval.x86_64.django_1776_django-12345"
    assert built["target"] == "source-minimal"


def test_swebench_image_template_overrides_official_image(monkeypatch):
    monkeypatch.setenv(
        "OPENHANDS_SWEBENCH_IMAGE_TEMPLATE",
        "ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id}:latest",
    )

    assert (
        build_images.get_official_docker_image("astropy__astropy-12907")
        == "ghcr.io/epoch-research/"
        "swe-bench.eval.x86_64.astropy__astropy-12907:latest"
    )
