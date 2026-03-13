import sys
from types import SimpleNamespace
from typing import Any, cast

from docker.errors import APIError, ImageNotFound

from benchmarks.swtbench.build_eval_env_images import (
    build_env_images,
    patch_swt_force_rebuild_remove_image,
)


def test_patch_swt_force_rebuild_remove_image_ignores_missing_local_image(
    monkeypatch,
):
    calls = []

    def original_remove_image(_client, image_id, logger=None):
        calls.append((image_id, logger))
        raise ImageNotFound("missing")

    docker_utils = SimpleNamespace(remove_image=original_remove_image)
    docker_build = SimpleNamespace(remove_image=original_remove_image)

    def fake_import_module(name):
        if name == "src.docker_utils":
            return docker_utils
        if name == "src.docker_build":
            return docker_build
        raise AssertionError(name)

    monkeypatch.setattr(
        "benchmarks.swtbench.build_eval_env_images.importlib.import_module",
        fake_import_module,
    )

    patch_swt_force_rebuild_remove_image()

    docker_utils.remove_image(object(), "exec.base.x86_64:latest", "quiet")

    assert calls == [("exec.base.x86_64:latest", "quiet")]
    assert docker_build.remove_image is docker_utils.remove_image


def test_patch_swt_force_rebuild_remove_image_preserves_other_errors(monkeypatch):
    boom = APIError("boom")

    def original_remove_image(_client, _image_id, _logger=None):
        raise boom

    docker_utils = SimpleNamespace(remove_image=original_remove_image)
    docker_build = SimpleNamespace(remove_image=original_remove_image)

    def fake_import_module(name):
        if name == "src.docker_utils":
            return docker_utils
        if name == "src.docker_build":
            return docker_build
        raise AssertionError(name)

    monkeypatch.setattr(
        "benchmarks.swtbench.build_eval_env_images.importlib.import_module",
        fake_import_module,
    )

    patch_swt_force_rebuild_remove_image()

    try:
        docker_utils.remove_image(object(), "exec.base.x86_64:latest", "quiet")
    except APIError as exc:
        assert exc is boom
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected docker.errors.APIError")


def test_build_env_images_force_build_avoids_upstream_force_rebuild(monkeypatch):
    remove_calls = []
    build_base_force_flags = []
    build_env_force_flags = []

    def original_remove_image(_client, image_id, logger=None):
        remove_calls.append((image_id, logger))
        raise ImageNotFound("missing")

    def fake_build_base_images(_client, dataset, force_rebuild=False, build_mode="api"):
        build_base_force_flags.append(force_rebuild)
        assert len(dataset) == 1

    def fake_build_env_images(
        _client,
        dataset,
        force_rebuild=False,
        max_workers=4,
        build_mode="api",
    ):
        build_env_force_flags.append(force_rebuild)
        assert len(dataset) == 2

    docker_utils = SimpleNamespace(remove_image=original_remove_image)
    docker_build = SimpleNamespace(
        BuildImageError=RuntimeError,
        build_base_images=fake_build_base_images,
        build_env_images=fake_build_env_images,
        remove_image=original_remove_image,
    )

    def fake_import_module(name):
        if name == "src.docker_utils":
            return docker_utils
        if name == "src.docker_build":
            return docker_build
        raise AssertionError(name)

    monkeypatch.setattr(
        "benchmarks.swtbench.build_eval_env_images.importlib.import_module",
        fake_import_module,
    )
    monkeypatch.setitem(sys.modules, "src", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "src.docker_build", docker_build)
    monkeypatch.setitem(sys.modules, "src.docker_utils", docker_utils)
    monkeypatch.setattr(
        "benchmarks.swtbench.build_eval_env_images.docker.from_env",
        lambda: object(),
    )

    spec = SimpleNamespace(
        base_image_key="exec.base.x86_64:latest",
        env_image_key="exec.env.foo:latest",
    )
    duplicate_spec = SimpleNamespace(
        base_image_key="exec.base.x86_64:latest",
        env_image_key="exec.env.foo:latest",
    )

    summary = build_env_images(
        exec_specs=[spec, duplicate_spec],
        max_workers=1,
        build_mode="cli",
        max_retries=0,
        batch_size=10,
        image_prefix=None,
        force_build=True,
    )

    assert build_base_force_flags == [False]
    assert build_env_force_flags == [False]
    assert remove_calls == [
        ("exec.base.x86_64:latest", "quiet"),
        ("exec.env.foo:latest", "quiet"),
    ]
    assert summary["built_base_images"] == 1
    assert summary["built_env_images"] == 1
    assert summary["selected_env_instances"] == 2
    assert summary["skipped_env_images"] == 0
    batches = cast(list[dict[str, Any]], summary["batches"])
    assert len(batches) == 1
    assert batches[0]["batch_index"] == 1
    assert batches[0]["batch_size"] == 1
    assert batches[0]["instance_count"] == 2
    assert batches[0]["attempt_count"] == 1
