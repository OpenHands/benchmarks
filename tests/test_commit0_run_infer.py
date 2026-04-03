"""Tests for commit0 run_infer test command helpers."""

from types import SimpleNamespace
from unittest.mock import sentinel

import pytest

from benchmarks.commit0 import build_images as commit0_build_images
from benchmarks.commit0.config import BUILD_TARGET
from benchmarks.commit0.run_infer import get_pythonpath_prefix, normalize_pytest_cmd
from benchmarks.utils.version import IMAGE_TAG_PREFIX


def test_commit0_defaults_to_source_minimal_target():
    assert BUILD_TARGET == "source-minimal"


def test_commit0_agent_server_image_tag_matches_run_infer():
    tag = commit0_build_images.get_agent_server_image_tag(
        "docker.io/wentingzhao/tinydb:v0",
        "source-minimal",
        "ghcr.io/example/agent-server",
    )

    assert (
        tag
        == "ghcr.io/example/agent-server:"
        f"{commit0_build_images.get_agent_server_image_tag_prefix('source-minimal')}"
        "-commit0-tinydb-source-minimal"
    )


def test_source_targets_include_agent_layer_hash_in_tag_prefix():
    prefix = commit0_build_images.get_agent_server_image_tag_prefix("source-minimal")

    assert prefix == (
        f"{IMAGE_TAG_PREFIX}-{commit0_build_images.agent_layer_content_hash()}"
    )


def test_source_targets_use_phased_assembly(monkeypatch):
    builder_calls = []
    assemble_calls = []

    def fake_builder_image(*, push, platform):
        builder_calls.append((push, platform))
        return SimpleNamespace(error=None, tags=["ghcr.io/example/builder:test"])

    def fake_assemble(**kwargs):
        assemble_calls.append(kwargs)
        return 0

    monkeypatch.setattr(commit0_build_images, "build_builder_image", fake_builder_image)
    monkeypatch.setattr(
        commit0_build_images, "assemble_commit0_agent_images", fake_assemble
    )

    exit_code = commit0_build_images.build_commit0_images(
        base_images=["docker.io/example/base:v0"],
        target="source-minimal",
        build_dir=sentinel.build_dir,
        image="ghcr.io/example/agent-server",
        push=True,
        max_workers=4,
        build_batch_size=2,
        dry_run=False,
        force_build=True,
        max_retries=3,
    )

    assert exit_code == 0
    assert builder_calls == [(True, "linux/amd64")]
    assert assemble_calls == [
        {
            "base_images": ["docker.io/example/base:v0"],
            "builder_tag": "ghcr.io/example/builder:test",
            "build_dir": sentinel.build_dir,
            "target_image": "ghcr.io/example/agent-server",
            "target": "source-minimal",
            "push": True,
            "max_workers": 4,
            "max_retries": 3,
            "force_build": True,
        }
    ]


def test_commit0_main_forwards_expected_build_args(monkeypatch):
    forwarded = {}

    monkeypatch.setattr(
        commit0_build_images,
        "collect_base_images",
        lambda **_: ["docker.io/example/base:v0"],
    )
    monkeypatch.setattr(
        commit0_build_images,
        "default_build_output_dir",
        lambda dataset, split: f"/tmp/{dataset}/{split}",
    )

    def fake_build_commit0_images(**kwargs):
        forwarded.update(kwargs)
        return 0

    monkeypatch.setattr(
        commit0_build_images, "build_commit0_images", fake_build_commit0_images
    )

    exit_code = commit0_build_images.main(
        [
            "--dataset",
            "dataset",
            "--split",
            "test",
            "--repo-split",
            "tinydb",
            "--image",
            "ghcr.io/example/agent-server",
            "--max-workers",
            "2",
            "--n-limit",
            "1",
        ]
    )

    assert exit_code == 0
    assert forwarded == {
        "base_images": ["docker.io/example/base:v0"],
        "target": "source-minimal",
        "build_dir": "/tmp/dataset/test",
        "image": "ghcr.io/example/agent-server",
        "push": False,
        "max_workers": 2,
        "build_batch_size": None,
        "dry_run": False,
        "force_build": False,
        "max_retries": 3,
    }


@pytest.mark.parametrize(
    "input_cmd, expected",
    [
        ("pytest", "python -m pytest"),
        ("pytest3", "python -m pytest3"),
        ("python -m pytest", "python -m pytest"),
        ("mypytest", "mypytest"),
        ("pytest-xdist", "pytest-xdist"),
        ("pytest_runner", "pytest_runner"),
        (
            "pytest --assert=plain --ignore=setup.py",
            "python -m pytest --assert=plain --ignore=setup.py",
        ),
    ],
    ids=[
        "bare_pytest",
        "bare_pytest3",
        "already_module_form",
        "substring_mypytest",
        "substring_pytest-xdist",
        "substring_pytest_runner",
        "real-parsel-scenario",
    ],
)
def test_normalize_pytest_cmd(input_cmd, expected):
    assert normalize_pytest_cmd(input_cmd) == expected


@pytest.mark.parametrize(
    "src_dir, expected",
    [
        ("src/cachetools", "PYTHONPATH=src:$PYTHONPATH "),
        ("src", "PYTHONPATH=src:$PYTHONPATH "),
        ("", ""),
        ("lib/mypackage", ""),
        ("tests/src/data", ""),
    ],
    ids=[
        "src_layout",
        "bare_src",
        "empty_string",
        "no_src_dir",
        "src_not_at_start",
    ],
)
def test_get_pythonpath_prefix(src_dir, expected):
    assert get_pythonpath_prefix(src_dir) == expected
