from types import SimpleNamespace

from docker.errors import APIError, ImageNotFound

from benchmarks.swtbench.build_eval_env_images import (
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
